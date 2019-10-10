#ifndef __FEWEST_SWITCHES_SURFACE_HOPPING_H__
#define __FEWEST_SWITCHES_SURFACE_HOPPING_H__

#include "TwoPara.h"

struct FSSH
{
	FSSH(	TwoPara* model_,
			double const& mass_,
			double const& dt_,
			arma::uword const& nt_);

	void initialize(double const& x0_, double const& v0_, bool const& state0_, double const& rho00_, std::complex<double> const& rho01);
	void propagate();
	void velocity_verlet();
	void quantum_Liouville();
	void stochastic_hop();
	void collect();
	double F(double const& x, bool const& state);
	double dc01(double const& x);
	arma::cx_vec drho_dt(arma::cx_vec const& rho);

	TwoPara* model;
	double mass;
	double x;
	double v;
	double a;
	double dt;
	bool state;
	double rho00;
	std::complex<double> rho01;

	arma::uword nt;
	arma::uword counter;
	
	arma::vec x_t;
	arma::vec v_t;
	arma::uvec state_t;
};

#endif
