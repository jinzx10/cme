#include "fssh.h"

double const DELTA = 1e-3;

FSSH::FSSH(	TwoPara* model_,
			double const& mass_,
			double const& dt_,
			arma::uword const& nt_):
	model(model_), mass(mass_), dt(dt_), nt(nt_),
	x_t(arma::zeros(nt)), v_t(arma::zeros(nt)), state_t(arma::zeros<arma::uvec>(nt))
{}

void FSSH::initialize(double const& x0_, double const& v0_, bool const& state0_, arma::cx_mat const& denmat0_) {
	x = x0_;
	v = v0_;
	state = state0_;
	counter = 0;

	a = F(x, state) / mass;
	denmat = denmat0_;

	x_t.zeros();
	v_t.zeros();
	state_t.zeros();

	collect();
}

void FSSH::velocity_verlet() {
	x += v * dt + 0.5 * a * dt * dt;
	double a_dt = F(x, state) / mass;
	v += 0.5 * (a + a_dt) * dt;
	a = a_dt;
}

void FSSH::quantum_Liouville() {
	
}

void FSSH::stochastic_hop() {
	arma::arma_rng::set_seed_random();
	arma::vec eigval = arma::eig_sym(model->H_dia(x));
	double DE = eigval(1) - eigval(0);
	if (state) { // excited -> ground
		double b01 = 2.0 * std::real( denmat(0,1) * v * dc01(x) ); // dc10 = -dc01
		if ( arma::randu() < b01*dt/denmat(1,1).real() ) {
			state = 0;
			v = std::sqrt( v*v + 2.0 * DE / mass );
		}
	} else { // ground -> excited
		double b10 = -2.0 * std::real( denmat(1,0) * v * dc01(x) );
		if ( arma::randu() < b10*dt/denmat(0,0).real() && 0.5 * mass * v * v > DE ) {
			state = 1;
			v = std::sqrt( v*v - 2.0 * DE / mass );
		}
	}
}

void FSSH::collect() {
	state_t(counter) = state;
	x_t(counter) = x;
	v_t(counter) = v;
}

void FSSH::propagate() {
	for (counter = 1; counter != nt; ++counter) {
		velocity_verlet();
		quantum_Liouville();
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


