#ifndef __FEWEST_SWITCHES_SURFACE_HOPPING_H__
#define __FEWEST_SWITCHES_SURFACE_HOPPING_H__

#include "TwoPara.h"
#include "DecayRate.h"

struct FSSH
{
	FSSH(	TwoPara*					model_,
			double			const&		mass_,
			double 			const& 		dt_,
			arma::uword		const& 		nt_,
			DecayRate		const&		Gamma_,
			double			const&		beta_		);

	void			initialize(bool const& state0_, double const& x0_, double const& v0_, double const& rho00_, std::complex<double> const& rho01_);
	void			propagate();

	// three tasks in one progagation step
	void			rk4_onestep();
	void 			hop();
	void 			collect();

	// first-order differential equation
	arma::vec		dvar_dt(arma::vec const& var_);

	TwoPara*		model;
	DecayRate		Gamma; // parameter in the additional damping term
	double			beta; // parameter in the additional damping term
	double 			mass;
	double 			dt;
	arma::uword 	nt;

	bool			state;
	arma::vec		var; // x, v, rho_00, Re(rho_01), Im(rho_01)
	arma::uword		counter;

	// data storage for one trajectory
	arma::vec		x_t;
	arma::vec		v_t;
	arma::vec		rho00_t;
	arma::vec		Re_rho01_t;
	arma::vec		Im_rho01_t;
	arma::uvec		state_t;
};

#endif
