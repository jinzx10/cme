#include "FSSH.h"

double const DELTA = 1e-3;
double const beta = 1.0 / 0.001;

FSSH::FSSH(		TwoPara*					model_,
				double			const&		mass_,
				double			const&		dt_,
				arma::uword		const& 		nt_,
				double			const&		Gamma_		):
	model(model_), Gamma(Gamma_), mass(mass_), dt(dt_), nt(nt_),
	x_t(arma::zeros(nt)), v_t(arma::zeros(nt)), state_t(arma::zeros<arma::uvec>(nt))
{}

void FSSH::initialize(bool const& state0_, double const& x0_, double const& v0_, double const& rho00_, std::complex<double> const& rho01_) {
	state = state0_;
	var = arma::zeros(5);
	var(0) = x0_;
	var(1) = v0_;
	var(2) = rho00_;
	var(3) = rho01_.real();
	var(4) = rho01_.imag();
	counter = 0;

	x_t.zeros();
	v_t.zeros();
	state_t.zeros();

	collect();
}

arma::vec FSSH::dvar_dt(arma::vec const& var_) {
	arma::vec dvar = arma::zeros(5);
	arma::vec eigval = arma::eig_sym( model->H_dia(var_(0)) );
	double E01 = eigval(0) - eigval(1);
	double rho00eq = 1.0 / ( 1.0 + std::exp(beta*E01) );
	double dc01_ = dc01(var_(0));

	return arma::vec{
		var_(1),
		F(var_(0)) / mass,
		-2.0 * var_(1) * dc01_ * var_(3) - Gamma * (var_(2) - rho00eq),
		E01 * var_(4) + var_(1) * dc01_ * (2.0*var_(2)-1.0) - Gamma / 2.0 * var_(3),
		-E01 * var_(3) - Gamma / 2.0 * var_(4)
	};
}

void FSSH::rk4_onestep() {
	arma::vec k1 = dt*dvar_dt(var);
	arma::vec k2 = dt*dvar_dt(var + 0.5*k1);
	arma::vec k3 = dt*dvar_dt(var + 0.5*k2);
	arma::vec k4 = dt*dvar_dt(var + k3);
	var += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
}

void FSSH::hop() {
	arma::arma_rng::set_seed_random();

	double x = var(0);
	double v = var(1);
	double rho00 = var(2);
	double drho00 = dvar_dt(var)(2);

	arma::vec eigval = arma::eig_sym(model->H_dia(x));
	double E10 = eigval(1) - eigval(0);

	if (state) { // excited -> ground
		if ( arma::randu() < (drho00 > 0) * dt * drho00 / (1.0-rho00) ) {
			state = 0;
			v = std::sqrt( v*v + 2.0 * E10 / mass );
		}
	} else { // ground -> excited
		if ( arma::randu() < (drho00 < 0) * dt * (-drho00) / rho00 ) {
			if ( 0.5 * mass * v * v > E10 ) {
				state = 1;
				v = std::sqrt( v*v - 2.0 * E10 / mass );
			} else {
				if ( F(x,1)*v < 0 ) v = -v;
			}
		}
	}
}

void FSSH::collect() {
	state_t(counter) = state;
	x_t(counter) = var(0);
	v_t(counter) = var(1);
}

void FSSH::propagate() {
	for (counter = 1; counter != nt; ++counter) {
		rk4_onestep();
		hop();
		collect();
#ifdef PRINT_PROGRESS
		std::cout << "x = " << var(0) << "   v = " << var(1) << "   state = " << state
			<< std::endl;
#endif
	}
}

double FSSH::F(double const& x) {
	arma::vec valm = arma::eig_sym( model->H_dia(x-DELTA) );
	arma::vec valp = arma::eig_sym( model->H_dia(x+DELTA) );
	return ( valm(state) - valp(state) ) / 2.0 / DELTA;
}

double FSSH::F(double const& x, bool const& state_) {
	arma::vec valm = arma::eig_sym( model->H_dia(x-DELTA) );
	arma::vec valp = arma::eig_sym( model->H_dia(x+DELTA) );
	return ( valm(state_) - valp(state_) ) / 2.0 / DELTA;
}

double FSSH::dc01(double const& x) {
	arma::mat DH = ( model->H_dia(x+DELTA) - model->H_dia(x-DELTA) ) / 2.0 / DELTA;
	arma::mat eigvec;
	arma::vec eigval;
	arma::eig_sym(eigval, eigvec, model->H_dia(x));
	return std::abs(arma::as_scalar(
				eigvec.col(0).t() * DH * eigvec.col(1) / (eigval(1) - eigval(0)) ) );
}


