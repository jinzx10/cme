#include "fssh.h"

double const DELTA = 1e-3;
std::complex<double> I(0.0, 1.0);
double const Gamma = 0.002;
double const beta = 1.0 / 0.001;

FSSH::FSSH(	TwoPara* model_,
			double const& mass_,
			double const& dt_,
			arma::uword const& nt_ ):
	model(model_), mass(mass_), dt(dt_), nt(nt_),
	x_t(arma::zeros(nt)), v_t(arma::zeros(nt)), state_t(arma::zeros<arma::uvec>(nt))
{}

void FSSH::initialize(bool const& state0_, double const& x0_, double const& v0_, double const& rho00_, std::complex<double> const& rho01_) {
	state = state0_;
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
	double x = var_(0);
	double v = var_(1);
	double rho00 = var_(2);
	double rho01R = var_(3);
	double rho01I = var_(4);

	arma::vec eigval = arma::eig_sym( model->H_dia(x) );
	double E01 = eigval(0) - eigval(1);
	double rho00eq = 1.0 / ( 1.0 + std::exp(beta*E01) );

	dvar(0) = v;
	dvar(1) = F(x, state) / mass;
	dvar(2) = -2.0 * v * dc01(x) * rho01R - Gamma * (rho00 - rho00eq);
	dvar(3) = E01*rho01I + v*dc01(x)*(2.0*rho00-1.0) - Gamma/2.0*rho01R;
	dvar(4) = -E01*rho01R - Gamma/2.0*rho01I;

	return dvar;
}

void FSSH::onestep() {
	arma::vec k1 = dt*dvar_dt(var);
	arma::vec k2 = dt*dvar_dt(var + 0.5*k1);
	arma::vec k3 = dt*dvar_dt(var + 0.5*k2);
	arma::vec k4 = dt*dvar_dt(var + k3);
	var += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
}

void FSSH::stochastic_hop() {
	arma::arma_rng::set_seed_random();
	double x = var(0);
	double v = var(1);
	double rho00 = var(2);
	double rho01R = var(3);
	arma::vec eigval = arma::eig_sym(model->H_dia(x));
	double DE = eigval(1) - eigval(0);
	if (state) { // excited -> ground
		double b01 = 2.0 * rho01R * v * dc01(x); // dc10 = -dc01
		if ( arma::randu() < b01*dt/(1.0-rho00) ) {
			state = 0;
			v = std::sqrt( v*v + 2.0 * DE / mass );
		}
	} else { // ground -> excited
		double b10 = -2.0 * rho01R * v * dc01(x);
		if ( arma::randu() < b10*dt/rho00 && 0.5 * mass * v * v > DE ) {
			state = 1;
			v = std::sqrt( v*v - 2.0 * DE / mass );
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
		onestep();
		stochastic_hop();
		collect();
	}
}

double FSSH::F(double const& x, bool const& state) {
	arma::vec valm = arma::eig_sym( model->H_dia(x-DELTA) );
	arma::vec valp = arma::eig_sym( model->H_dia(x+DELTA) );
	return ( valm(state) - valp(state) ) / 2.0 / DELTA;
}

double FSSH::dc01(double const& x) {
	arma::mat DH = ( model->H_dia(x+DELTA) - model->H_dia(x-DELTA) ) / 2.0 / DELTA;
	arma::mat eigvec;
	arma::vec eigval;
	arma::eig_sym(eigval, eigvec, model->H_dia(x));
	return std::abs(arma::as_scalar(
				eigvec.col(0) * DH * eigvec.col(1) / (eigval(1) - eigval(0)) ) );
}


