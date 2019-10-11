#ifndef __FEWEST_SWITCHES_SURFACE_HOPPING_H__
#define __FEWEST_SWITCHES_SURFACE_HOPPING_H__

#include "TwoPara.h"

struct FSSH
{
	FSSH(	TwoPara* model_,
			double const& mass_,
			double const& dt_,
			arma::uword const& nt_);

	void initialize(bool const& state0_, double const& x0_, double const& v0_, double const& rho00_, std::complex<double> const& rho01);
	void propagate();
	void onestep();
	arma::vec dvar_dt(arma::vec const& var_);
	void stochastic_hop();
	void collect();

	double F(double const& x, bool const& state);
	double dc01(double const& x);
	arma::cx_vec drho_dt(arma::cx_vec const& rho);



	TwoPara* model;
	double mass;
	double dt;
	arma::uword nt;

	bool state;
	arma::vec var; // x, v, rho_00, Re(rho_01), Im(rho_01)
	arma::uword counter;

	arma::vec x_t;
	arma::vec v_t;
	arma::uvec state_t;
};

#endif
