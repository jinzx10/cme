#include "Langevin.h"

Langevin::Langevin(		TwoPara*					model_,
						double			const&		mass_,
						double			const&		dt_,
						arma::uword		const&		nt_,
						double			const&		beta_,
						double			const&		gamma_		):
	model(model_), mass(mass_), dt(dt_), nt(nt_), beta(beta_), gamma(gamma_)
{
	var = arma::zeros(2);
	counter = 0;
	x_t = arma::zeros(nt);
	v_t = arma::zeros(nt);
}

void Langevin::initialize(double const& x0_, double const& v0_) {
	var(0) = x0_;
	var(1) = v0_;
	counter = 0;
	x_t.zeros();
	v_t.zeros();
	collect();
}

double Langevin::a() {
	return ( model->F(var(0), 0) 
			- gamma * var(1) 
			+ arma::randn() * std::sqrt(2*gamma/beta/dt) ) / mass;
}

void Langevin::velocity_verlet() {
	double a_old = a();
	var(0) += var(1) * dt + 0.5 * a_old * dt * dt;
	double a_new = a();
	var(1) += 0.5 * (a_old + a_new) * dt;
}

void Langevin::propagate() {
	for (counter = 1; counter != nt; ++counter) {
		velocity_verlet();
		collect();
	}
}

void Langevin::collect() {
	x_t(counter) = var(0);
	v_t(counter) = var(1);
}

