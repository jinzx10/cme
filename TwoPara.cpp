#include "TwoPara.h"

TwoPara::TwoPara(PES const& E_mpt_, PES const& E_fil_, arma::vec const& bath_, arma::vec const& cpl_, arma::uword const& n_occ_):
	E_mpt(E_mpt_), E_fil(E_fil_), bath(bath_), cpl(cpl_), n_occ(n_occ_)
{}

arma::uword TwoPara::n_bath() {
	return bath.n_elem;
}

arma::mat TwoPara::H_elec(double const& x) {
	arma::mat H = arma::zeros(n_bath()+1, n_bath()+1);
	H.diag() = arma::join_cols( arma::vec{E_fil(x) - E_mpt(x)}, bath );
	H(0, arma::span(1, n_bath())) = cpl.st();
	H(arma::span(1, n_bath()), 0) = cpl;
	return H;
}

arma::mat TwoPara::H_dia(double const& x) {
	arma::mat H = arma::zeros(2,2);
	arma::mat eigvec;
	arma::vec eigval;
	arma::eig_sym(eigval, eigvec, H_elec(x));

	double E0 = arma::sum( eigval(arma::span(0, n_occ-1)) );
	double n0 = arma::sum( arma::square(eigvec(0, arma::span(0, n_occ-1))) );
	//double Ev = arma::sum( eigval(arma::span(n_occ, n_bath())) );
	double nv = arma::sum( arma::square(eigvec(0, arma::span(n_occ, n_bath()))) );
	double n0E0 = arma::as_scalar( arma::square(eigvec(0, arma::span(0, n_occ-1))) *
			eigval(arma::span(0, n_occ-1)) );
	double nvEv = arma::as_scalar( arma::square(eigvec(0, arma::span(n_occ, n_bath()))) *
			eigval(arma::span(n_occ, n_bath())) );
	double nH = n0 * E0;
	double nHn = n0*nv*E0 - nv*n0E0 + n0*nvEv + n0*n0*E0;

	H(0,0) = (E0 + nHn - nH*2) / (1-n0);
	H(0,1) = (nH - nHn) / sqrt(n0*(1-n0));
	H(1,0) = H(0,1);
	H(1,1) = nHn / n0;
	return H;
}
