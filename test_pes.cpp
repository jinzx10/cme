#include "TwoPara.h"

double const pi = acos(-1.0);

int main() {
	double x0_mpt = 2;
	double x0_fil = 3;
	double omega_mpt = 0.04;
	double omega_fil = 0.04;
	double mass = 7;
	double dE_fil = 0.001;

	auto E_mpt = [&](double const& x) { return 0.5 * mass * omega_mpt * omega_mpt * 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&](double const& x) { return 0.5 * mass * omega_fil * omega_fil * 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};

	double bath_width = 0.04;

	double Gamma = 0.002;

	TwoPara model(E_mpt, E_fil, Gamma, 0.0, bath_width);

	arma::uword nx = 100;
	arma::vec xgrid = arma::linspace(x0_mpt-0.3, x0_fil+0.3, nx);

	arma::mat H_store = arma::zeros(nx, 5);

	std::cout << "begin:" << std::endl;
	for (arma::uword ix = 0; ix != nx; ++ix) {
		double x = xgrid(ix);
		arma::mat H_tmp = model.H_dia(x);
		H_store(ix, 0) = x;
		H_store(ix, 1) = H_tmp(0,0);
		H_store(ix, 2) = H_tmp(1,1);
		H_store(ix, 3) = H_tmp(0,1);
		H_store(ix, 4) = E_mpt(x);
		std::cout << "\e[A" << ix+1 << "/" << nx << std::endl;
	}

	H_store.save("H_dia.txt", arma::raw_ascii);

	return 0;
}
