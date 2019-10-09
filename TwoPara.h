#ifndef __TWO_PARABOLA_H__
#define __TWO_PARABOLA_H__

#include <armadillo>
#include <functional>

struct TwoPara
{
	using PES = std::function<double(double)>;

	TwoPara(PES const& E_mpt_,
			PES const& E_fil_,
			arma::vec const& bath_,
			arma::vec const& cpl_,
			arma::uword const& n_occ_);


	PES E_mpt;
	PES E_fil;

	arma::vec bath;
	arma::vec cpl;
	arma::uword n_occ;

	arma::uword n_bath();

	arma::mat H_elec(double const& x);
	arma::mat H_dia(double const& x);
};

#endif
