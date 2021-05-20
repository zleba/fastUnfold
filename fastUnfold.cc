#include <iostream>
#include <cmath>
#include <chrono>

#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Inputs
// y    : detLevel data
// Vyy  : detLevel data Cov

// M    : Migration matrix

// x0   : True-level bias
// L    : The regularization matrix


struct fastUnfold { 

    VectorXd y;    // detLevel data
    MatrixXd Vyy;  // detLevel data Cov

    MatrixXd M;    // Migration matrix
    MatrixXd A;    // Response matrix

    VectorXd x0;  // True-level bias
    MatrixXd L;   // The regularization matrix


    // Outut
    VectorXd x;
    MatrixXd Vxx;


    void Unfold(double tau)
    {

        MatrixXd     m  = A.transpose() * Vyy.inverse(); //can be done as linear-system solution
        MatrixXd     LL = tau*tau * L * L.transpose();
        MatrixXd Einv = m*A + LL;

        MatrixXd E = Einv.inverse();

        // resulting true-level vector
        x = E * ( m*y + LL * x0);

        MatrixXd Dxy = E * m;

        // resulting true-level covariance
        Vxx = Dxy * Vyy * Dxy.transpose();

    }


};



#include "TUnfold.h"

TString rn() { return Form("%d", rand()); }

TH1D *toHist(const VectorXd &vec)
{
    TH1D *h = new TH1D(rn(), "", vec.cols(), -0.5, -0.5 + vec.cols() );
    for(int i = 0; i < vec.cols(); ++i) {
        h->SetBinContent(i+1, vec[i]);
        h->SetBinError(i+1, 0);
    }
    return h;
}

TH2D *toHist(const MatrixXd &mat)
{
    TH2D *h = new TH2D(rn(), "", mat.rows(), -0.5, -0.5 + mat.rows(), mat.cols(), -0.5, -0.5 + mat.cols()  );
    for(int i = 0; i < mat.rows(); ++i)
        for(int j = 0; j < mat.cols(); ++j) {
            h->SetBinContent(i+1, j+1, mat(i,j));
            h->SetBinError(i+1, j+1, 0);
        }
    return h;
}


void TUnfoldWay(MatrixXd A, VectorXd y, MatrixXd yy, double tau)
{

    TH2D *hA  = toHist(A);
    TH1D *hy  = toHist(y);
    TH2D *hyy = toHist(yy);

    TUnfold *unf = new TUnfold(hA, TUnfold::EHistMap::kHistMapOutputVert,
                               TUnfold::ERegMode::kRegModeSize, TUnfold::EConstraint::kEConstraintNone);

    unf->SetInput(hy, /*scaleBias*/0.0, /*oneOverZeroError*/0.0, hyy);

    unf->DoUnfold(tau);
    
    TH1D *hx = new TH1D(rn(), "", unf->GetNpar(), -0.5, -0.5 + unf->GetNpar());
    unf->GetOutput(hx);
    cout << "Ngen " << unf->GetNpar() << endl;
    //for(int i = 1; i < hx->GetNbinsX(); ++i)
        //cout << i << " " << hx->GetBinContent(i) << endl;


}

double gaus(double x)
{
    const double sigma = 5;
    return exp(-x*x/sigma/sigma/2.) * (1 + 0.3*rand()/(RAND_MAX+0));
}

int main()
{
    //Fill response matrix
    int n = 600;
    MatrixXd M(2*n, n);
    for(int j = 0; j < n; ++j) { //gen loop
        double s = 0;
        for(int i = 0; i < 2*n; ++i) { //rec loop
            M(i,j) = gaus(i - 2*j);
            s += M(i,j);
        }

        for(int i = 0; i < 2*n; ++i)
            M(i,j) /= s;
    }

    VectorXd vX(n);
    for(int i = 0; i < n; ++i)
        vX(i) = i + 3;


    VectorXd vY = M * vX;

    MatrixXd Vyy(2*n, 2*n);
    for(int i = 0; i < 2*n; ++i)
        for(int j = i; j < 2*n; ++j) {
            if(i == j)
                Vyy(i,j) = 1;
            else
                Vyy(i,j) = Vyy(j,i) =  (0.02*rand()) / (RAND_MAX+0.);
        }

    //cout << "Rank Vyy " << Vyy.determinant() << endl;
    

    fastUnfold fu;
    fu.A = M;
    fu.y = vY;
    fu.Vyy= Vyy;
    fu.x0 = vX * 0;
    fu.L  = vX * 0;
    for(int i = 0; i < fu.L.rows(); ++i) fu.L(i) = 1;


    cout << "Fast unf: start" << endl;
    auto begin = std::chrono::steady_clock::now();
    fu.Unfold(1e-5);
    auto end = std::chrono::steady_clock::now();
    cout << "Fast unf: end : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << endl;



    cout << "TUnfolding: start" << endl;
    begin = std::chrono::steady_clock::now();
    TUnfoldWay(fu.A, fu.y, fu.Vyy, 1e-5);
    end = std::chrono::steady_clock::now();
    cout << "TUnfolding: end : "  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << endl;

    return 0;
}
