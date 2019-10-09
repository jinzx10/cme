#ifndef __FEWEST_SWITCHES_SURFACE_HOPPING_H__
#define __FEWEST_SWITCHES_SURFACE_HOPPING_H__

#include "TwoPara.h"

struct FSSH
{
	FSSH(	TwoPara* model_,
			double const& mass_,
			double const& dt_,
			arma::uword const& nt_);

	void initialize(double const& x0_, double const& v0_, bool const& state0_, arma::cx_mat const& denmat0_);
	void propagate();
	void velocity_verlet();
	void quantum_Liouville();
	void stochastic_hop();
	void collect();
	double F(double const& x, bool const& state);
	double dc01(double const& x);

	TwoPara* model;
	double mass;
	double x;
	double v;
	double a;
	double dt;
	bool state;
	arma::cx_mat denmat;

	arma::uword nt;
	arma::uword counter;
	
	arma::vec x_t;
	arma::vec v_t;
	arma::uvec state_t;
};

#endif
