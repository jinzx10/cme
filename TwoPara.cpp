#include "TwoPara.h"

TwoPara::TwoPara(
		PES				const&		E_mpt_,
		PES				const& 		E_fil_,
		double			const&		Gamma_, 
		double			const&		E_fermi_,
		double			const&		W_band_		):
	E_mpt(E_mpt_), E_fil(E_fil_), Gamma(Gamma_), E_fermi(E_fermi_), W_band(W_band_)
{
	double n_grid = (E_fermi + W_band) / Gamma * 30;
	E_grid = arma::linspace(-W_band, E_fermi, n_grid);
	E_grid.shed_row(0);
	dE_grid = E_grid(1) - E_grid(0);
	Re_Self = Gamma / 2.0 / arma::datum::pi *
		arma::log( ( W_band + E_grid ) / ( W_band - E_grid ) );
}


double TwoPara::Ed(double const& x) {
	return E_fil(x) - E_mpt(x);
}


double TwoPara::ev_n(double const& x) {
	return dE_grid * arma::sum( Gamma / 2.0 / arma::datum::pi /
			( arma::square(E_grid - Ed(x) - Re_Self) + Gamma*Gamma/4.0 ) );
}


double TwoPara::ev_H(double const& x) { // bath contribution is deducted
	return dE_grid * arma::sum( Gamma / 2.0 / arma::datum::pi * E_grid /
			( arma::square(E_grid - Ed(x) - Re_Self) + Gamma*Gamma/4.0 ) );
}


arma::mat TwoPara::H_dia(double const& x) {
	arma::mat H = arma::zeros(2,2);

	double n = ev_n(x);
	double E = ev_H(x);
	double nH1n = E - Ed(x)*n;
	double nHn = n*E - nH1n;

	H(0,1) = nH1n / std::sqrt(n*(1.0-n));
	H(1,0) = H(0,1);
	H(1,1) = nHn / n + E_mpt(x);
	H(0,0) = (E - 2.0*n*E + nHn) / (1.0-n) + E_mpt(x);

	return H;
}

double TwoPara::F(double const& x, bool const& state) {
	arma::vec valm = arma::eig_sym( H_dia(x-DELTA) );
	arma::vec valp = arma::eig_sym( H_dia(x+DELTA) );
	return ( valm(state) - valp(state) ) / 2.0 / DELTA;
}

double TwoPara::dc01(double const& x) {
	arma::mat DH = ( H_dia(x+DELTA) - H_dia(x-DELTA) ) / 2.0 / DELTA;
	arma::mat eigvec;
	arma::vec eigval;
	arma::eig_sym(eigval, eigvec, H_dia(x));
	return std::abs(arma::as_scalar(
				eigvec.col(0).t() * DH * eigvec.col(1) / (eigval(1) - eigval(0)) ) );
}

