#include "newtonroot.h"

double f(double x) {
	return 3.*x*x-2*x-3;
}

arma::vec fv(arma::vec v) {
	return arma::vec{v(0)-2*v(1), 3*v(0)*v(0)-5};
}

int main() {
	double x0 = newtonroot(f, 1.5);

	std::cout << x0 << std::endl;
	std::cout << f(x0) << std::endl;

	arma::vec v0 = newtonroot(fv, arma::vec{1,1});
	v0.print();
	fv(v0).print();

	return 0;
}
