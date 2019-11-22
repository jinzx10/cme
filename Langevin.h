#ifndef __LANGEVIN_DYNAMICS_H__
#define __LANGEVIN_DYNAMICS_H__

#include "TwoPara.h"

struct Langevin
{
	Langevin(	TwoPara*					model_,
				double		const&			mass_,
				double		const&			dt_,
				arma::uword		const&		nt_,
				double			const&		beta_,
				double			const&		gamma_		);

	void			initialize(double const& x0_, double const& v0_);
	void			propagate();

	double			a();
	void			velocity_verlet();
	void			collect();

	TwoPara*		model;
	double 			mass;
	double 			dt;
	arma::uword 	nt;
	double			beta; // inverse temperature
	double			gamma; // friction coefficient

	arma::vec		var; // x, v
	arma::uword		counter;

	// data storage for one trajectory
	arma::vec		x_t;
	arma::vec		v_t;
};

#endif
