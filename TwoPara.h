#ifndef __TWO_PARABOLA_H__
#define __TWO_PARABOLA_H__

#include <armadillo>
#include <functional>

struct TwoPara
{
	using PES = std::function<double(double)>;

	TwoPara(	PES				const&		E_mpt_,
				PES				const& 		E_fil_,
				double			const& 		Gamma_,
				double			const&		E_fermi_,
				double			const&		W_band_		);


	PES				E_mpt;
	PES				E_fil;

	double			Gamma;
	double			E_fermi;
	double			W_band;

	double			dE_grid;
	arma::vec		E_grid;
	arma::vec		Re_Self;

	double			Ed(double const& x);
	double			ev_n(double const& x);
	double			ev_H(double const& x);

	arma::mat		H_dia(double const& x);
};

#endif
