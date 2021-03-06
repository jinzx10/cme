#ifndef __SINGLE_IMPURITY_ANDERSON_MODEL_H__
#define __SINGLE_IMPURITY_ANDERSON_MODEL_H__

#include <functional>
#include <armadillo>

struct SIAM
{
	using d2d = std::function<double(double)>;
	using d2v = std::function<arma::vec(double)>;

	SIAM(	d2d				const&		Ed,
			d2v				const&		cpl,
			double			const&		U,
			arma::vec		const&		bath,
			arma::uword		const&		n_elec,
			arma::uword		const&		n_val		);

	// impurity on-site energy and impurity-bath coupling amplitudes
	// both are functions of the nuclear coordinate (which is a double)
	d2d						Ed;
	d2v						cpl;

	double					U;
	arma::vec				bath;
	arma::uword				n_elec;
	arma::uword				n_val;

	arma::uword				n_bath;
	arma::uword				n_occ;
	arma::uword				n_vir;
	arma::span				idx_F_occ;
	arma::span				idx_F_vir;

	arma::uword				max_scf_cycles = 50;
	double					tol_n_imp = 1e-10;

	double					E_mf = -1;
	double					n_imp = -1;
	arma::mat				vec_Fock;
	arma::vec				val_Fock;

	double					E_eff(double const& x, double const& n);
	arma::mat				Fock(double const& x, double const& n);
	double					n2n(double const& x, double const& n);
	void					mfscf(double const& x);

	void					CIS_common();

	arma::vec				vec_h;
	arma::vec				vec_l;
	arma::mat				Ftoo;
	arma::mat				Ftvv;

	arma::subview<double>	vec_Fock_occ();
	arma::subview<double>	vec_Fock_vir();

	arma::mat				Q_occ();
	arma::mat				Q_vir();


};

#endif
