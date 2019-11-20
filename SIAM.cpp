#include "SIAM.h"
#include <iostream>

SIAM::SIAM(		d2d				const&		Ed_,
				d2v 			const& 		cpl_,
				double			const& 		U_, 
				arma::vec		const& 		bath_,
				arma::uword		const&		n_elec_,
				arma::uword 	const& 		n_val_			):
	Ed(Ed_), cpl(cpl_), U(U_), bath(bath_), n_elec(n_elec_), n_val(n_val_)
{
	n_bath = bath_.n_elem;
	idx_F_occ = arma::span(0, n_bath/2-1);
	idx_F_vir = arma::span(n_bath/2, n_bath);
}

double SIAM::E_eff(double const& x, double const& n) {
	return Ed(x) + U * n;
}

arma::mat SIAM::Fock(double const& x, double const& n) {
	arma::mat F = arma::diagmat( arma::join_cols(bath, arma::vec{E_eff(x,n)}) );
	F(arma::span(0, n_bath-1), n_bath) = cpl(x);
	F(n_bath, arma::span(0, n_bath-1)) = F(arma::span(0, n_bath-1), n_bath).t();
	return F;
}

double SIAM::n2n(double const& x, double const& n) {
	arma::mat eigvec;
	arma::vec eigval;
	arma::eig_sym(eigval, eigvec, Fock(x,n));
	return arma::accu(arma::square( eigvec(n_bath, idx_F_occ) ));
}

void SIAM::mfscf(double const& x) {
#ifdef READ_LAST_N
	double n_imp_old = (n_imp < 0) ? ( Ed(x) < bath(n_elec/2) ) : n_imp;
#else
	double n_imp_old = ( Ed(x) < bath(n_elec/2) );
#endif

	double n_imp_new = n2n(x, n_imp_old);
	double g_old = n_imp_new - n_imp_old;
	double g_new = 0.0, dg = 0.0;

	arma::uword counter = 0;
	while ( std::abs(n_imp_old - n_imp_new) > tol_n_imp ) {
		if (counter > n_scf) {
			std::cout << "x = " << x << " mean-field SCF fails to converge." << std::endl;
			break;
		}

		g_new = n2n(x, n_imp_new) - n_imp_new;
        dg = (g_new - g_old) / (n_imp_new - n_imp_old);
        n_imp_old = n_imp_new;
        n_imp_new = n_imp_new - g_new / dg;
        g_old = g_new;

		counter += 1;
	}

	n_imp = n_imp_new;
	arma::eig_sym(val_Fock, vec_Fock, Fock(x,n_imp));
	E_mf = 2 * arma::accu(val_Fock(idx_F_occ)) - U * n_imp * n_imp;
}

arma::subview<double> SIAM::vec_Fock_occ() {
	return vec_Fock(arma::span::all, idx_F_occ);
}

arma::subview<double> SIAM::vec_Fock_vir() {
	return vec_Fock(arma::span::all, idx_F_vir);
}


