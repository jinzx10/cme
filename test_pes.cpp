#include <string>
#include <cstdlib>
#include "TwoPara.h"

double const pi = acos(-1.0);

int main() {
	double x0_mpt = 2;
	double x0_fil = 2.3;
	double omega = 0.002;
	double mass = 14000;
	double dE_fil = 0.0005;
	std::string datadir = "/home/zuxin/job/cme/data/pes/";
	std::string cmd = "mkdir -p " + datadir;

	auto E_mpt = [&](double const& x) { return 0.5 * mass * omega * omega * 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&](double const& x) { return 0.5 * mass * omega * omega * 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};

	double bath_width = 0.05;

	double Gamma = 0.0001;
	double E_fermi = 0.0;

	TwoPara model(E_mpt, E_fil, Gamma, E_fermi, bath_width);

	arma::uword nx = 100;
	arma::vec xgrid = arma::linspace(x0_mpt-0.3, x0_fil+0.5, nx);

	arma::mat H_store = arma::zeros(nx, 7);
	arma::mat Eg_store = arma::zeros(nx,2);
	arma::mat n_store = arma::zeros(nx,2);

	std::cout << "begin:" << std::endl;
	for (arma::uword ix = 0; ix != nx; ++ix) {
		double x = xgrid(ix);
		arma::mat H_tmp = model.H_dia(x);
		arma::vec eigval = arma::eig_sym(H_tmp);

		H_store(ix, 0) = x;
		H_store(ix, 1) = H_tmp(0,0);
		H_store(ix, 2) = H_tmp(1,1);
		H_store(ix, 3) = H_tmp(0,1);
		H_store(ix, arma::span(4,5)) = eigval.t();
		H_store(ix, 6) = E_mpt(x);

		Eg_store(ix,0) = x;
		Eg_store(ix,1) = model.ev_H(x);

		n_store(ix,0) = x;
		n_store(ix,1) = model.ev_n(x);

		std::cout << "\e[A" << ix+1 << "/" << nx << std::endl;
	}

	std::system(cmd.c_str());
	H_store.save(datadir+"pes.txt", arma::raw_ascii);
	Eg_store.save(datadir+"Eg.txt", arma::raw_ascii);
	n_store.save(datadir+"n.txt", arma::raw_ascii);

	return 0;
}
