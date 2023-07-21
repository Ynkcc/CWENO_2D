// mytest2.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
// CWENO_2D.cpp: 定义应用程序的入口点。
//
#include "CWENO_2D.h"
#include <iostream>
#include <armadillo>
#include <string>
using namespace arma;
using namespace std;

//测试通过 7*7矩阵
void deriv_reconstruct(mat& u, mat& ucx, mat& ucxx, mat& ucy, mat& ucyy, mat& ucxy,
    mat& BNE, mat& BNW, mat& BSE, mat& BSW, mat& CNE, mat& CNW, mat& CSE, mat& CSW)
{
    int n = u.n_rows;
    int m = u.n_cols;

    ucx.resize(n, m);
    ucxx.resize(n, m);
    ucy.resize(n, m);
    ucyy.resize(n, m);
    ucxy.resize(n, m);
    BNE.resize(n, m);
    BNW.resize(n, m);
    BSE.resize(n, m);
    BSW.resize(n, m);
    CNE.resize(n, m);
    CNW.resize(n, m);
    CSE.resize(n, m);
    CSW.resize(n, m);

    // calculate ucx and ucxx
    for (int j = 0; j < m; j++)
    {
        for (int i = 1; i < n - 1; i++)
        {
            ucx(i, j) = (u(i + 1, j) - u(i - 1, j)) / 2;
            ucxx(i, j) = u(i + 1, j) - 2 * u(i, j) + u(i - 1, j);
        }
    }
    ucx.row(0) = (u.row(1) - u.row(n - 2)) / 2;
    ucxx.row(0) = u.row(1) - 2 * u.row(0) + u.row(n - 2);
    ucx.row(n - 1) = ucx.row(0);
    ucxx.row(n - 1) = ucxx.row(0);

    // calculate ucy and ucyy
    for (int i = 0; i < n; i++)
    {
        for (int j = 1; j < m - 1; j++)
        {
            ucy(i, j) = (u(i, j + 1) - u(i, j - 1)) / 2;
            ucyy(i, j) = u(i, j + 1) - 2 * u(i, j) + u(i, j - 1);
        }
    }
    ucy.col(0) = (u.col(1) - u.col(m - 2)) / 2;
    ucyy.col(0) = u.col(1) - 2 * u.col(0) + u.col(m - 2);
    ucy.col(m - 1) = ucy.col(0);
    ucyy.col(m - 1) = ucyy.col(0);

    // calculate ucxy
    for (int i = 1; i < n - 1; i++)
    {
        for (int j = 1; j < m - 1; j++)
        {
            ucxy(i, j) = (u(i + 1, j + 1) + u(i - 1, j - 1) - u(i + 1, j - 1) - u(i - 1, j + 1)) / 4;
        }
    }

    // Compute ucxy
    for (int j = 1; j < m - 1; ++j) {
        ucxy(0, j) = (u(1, j + 1) + u(n - 2, j - 1) - u(1, j - 1) - u(n - 2, j + 1)) / 4.0;
    }
    for (int i = 1; i < n - 1; ++i) {
        //将m-1修改了，原来是自动生成
        ucxy(i, 0) = (u(i + 1, 1) + u(i - 1, m - 2) - u(i + 1, m - 2) - u(i - 1, 1)) / 4.0;
    }
    ucxy(0, 0) = (u(1, 1) + u(n - 2, m - 2) - u(1, m - 2) - u(n - 2, 1)) / 4.0;
    ucxy(0, m - 1) = ucxy(0, 0);
    ucxy(n - 1, 0) = ucxy(0, 0);
    ucxy(n - 1, m - 1) = ucxy(0, 0);

    // Compute BNE and BSE
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n - 1; ++i) {
            BNE(i, j) = u(i + 1, j) - u(i, j);
        }
    }
    BNE.row(n - 1) = BNE.row(0);
    BSE = BNE;

    // Compute BNW and BSW
    for (int j = 0; j < m; ++j) {
        for (int i = 1; i < n; ++i) {
            BNW(i, j) = u(i, j) - u(i - 1, j);
        }
    }
    BNW.row(0) = BNW.row(n - 1);
    BSW = BNW;

    // Compute CNE and CNW
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m - 1; ++j) {
            CNE(i, j) = u(i, j + 1) - u(i, j);
        }
    }
    CNE.col(m - 1) = CNE.col(0);
    CNW = CNE;

    for (int i = 0; i < n; ++i) {
        for (int j = 1; j < m; ++j) {
            CSW(i, j) = u(i, j) - u(i, j - 1);
        }
    }
    CSW.col(0) = CSW.col(m - 1);
    CSE = CSW;
}


void weight2(arma::mat& omegaNE, arma::mat& omegaNW, arma::mat& omegaSE, arma::mat& omegaSW, arma::mat& omegaC,
    const arma::mat& u, const arma::mat& ucx, const arma::mat& ucxx, const arma::mat& ucy, const arma::mat& ucyy, const arma::mat& ucxy,
    const arma::mat& BNE, const arma::mat& BNW, const arma::mat& BSE, const arma::mat& BSW, const arma::mat& CNE, const arma::mat& CNW, const arma::mat& CSE, const arma::mat& CSW) {
    int n = u.n_rows;
    int m = u.n_cols;
    omegaNE.resize(n, m);
    omegaNW.resize(n, m);
    omegaSE.resize(n, m);
    omegaSW.resize(n, m);
    omegaC.resize(n, m);
    arma::mat ISNE = BNE % BNE + CNE % CNE;
    arma::mat ISNW = BNW % BNW + CNW % CNW;
    arma::mat ISSW = BSW % BSW + CSW % CSW;
    arma::mat ISSE = BSE % BSE + CSE % CSE;
    arma::mat ISC = ucx % ucx + ucy % ucy + 1.0 / 3.0 * (13 * ucxx % ucxx + 14 * ucxy % ucxy + 13 * ucyy % ucyy);

    double CkNE = 1.0 / 8.0;
    double CkNW = 1.0 / 8.0;
    double CkSE = 1.0 / 8.0;
    double CkSW = 1.0 / 8.0;
    double CkC = 1.0 / 2.0;
    double xi = 1e-4;
    int p = 2;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double temp = 0.0;
            double alpha_NE = CkNE / pow(xi + ISNE(i, j), p);
            double alpha_NW = CkNW / pow(xi + ISNW(i, j), p);
            double alpha_SE = CkSE / pow(xi + ISSE(i, j), p);
            double alpha_SW = CkSW / pow(xi + ISSW(i, j), p);
            double alpha_C = CkC / pow(xi + ISC(i, j), p);

            temp = alpha_NE + alpha_NW + alpha_SE + alpha_SW + alpha_C;
            omegaNE(i, j) = alpha_NE / temp;
            omegaNW(i, j) = alpha_NW / temp;
            omegaSE(i, j) = alpha_SE / temp;
            omegaSW(i, j) = alpha_SW / temp;
            omegaC(i, j) = alpha_C / temp;
        }
    }
}


//测试通过 7*7矩阵
void Rjx2(arma::mat u, int option, arma::mat& I1, arma::mat& Ajk, arma::mat& omegaNE, arma::mat& omegaNW, arma::mat& omegaSE, arma::mat& omegaSW, arma::mat& omegaC) {
    int n = u.n_rows;
    int m = u.n_cols;
    arma::mat ucx(n, m), ucxx(n, m), ucy(n, m), ucyy(n, m), ucxy(n, m);
    arma::mat BNE(n, m), BNW(n, m), BSE(n, m), BSW(n, m);
    arma::mat CNE(n, m), CNW(n, m), CSE(n, m), CSW(n, m);

    deriv_reconstruct(u, ucx, ucxx, ucy, ucyy, ucxy, BNE, BNW, BSE, BSW, CNE, CNW, CSE, CSW);
    weight2(omegaNE, omegaNW, omegaSE, omegaSW, omegaC, u, ucx, ucxx, ucy, ucyy, ucxy, BNE, BNW, BSE, BSW, CNE, CNW, CSE, CSW);

    Ajk.zeros(n, m);
    arma::mat Bjk(n, m);
    arma::mat Cjk(n, m);
    arma::mat Djk(n, m);
    arma::mat Ejk(n, m);
    arma::mat Fjk(n, m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            Ajk(i, j) = omegaNE(i, j) * u(i, j) + omegaNW(i, j) * u(i, j) + omegaSW(i, j) * u(i, j) + omegaSE(i, j) * u(i, j)
                + omegaC(i, j) * (u(i, j) - 1.0 / 12 * (ucxx(i, j) + ucyy(i, j)));

            Bjk(i, j) = omegaNE(i, j) * BNE(i, j) + omegaNW(i, j) * BNW(i, j) + omegaSW(i, j) * BSW(i, j)
                + omegaSE(i, j) * BSE(i, j) + omegaC(i, j) * ucx(i, j);

            Cjk(i, j) = omegaNE(i, j) * CNE(i, j) + omegaNW(i, j) * CNW(i, j) + omegaSW(i, j) * CSW(i, j)
                + omegaSE(i, j) * CSE(i, j) + omegaC(i, j) * ucy(i, j);

            Djk(i, j) = omegaC(i, j) * 2 * ucxy(i, j);
            Ejk(i, j) = omegaC(i, j) * ucxx(i, j);
            Fjk(i, j) = omegaC(i, j) * ucyy(i, j);
        }
    }
    I1.zeros(n, m);
    switch (option) {
    case 0: {
        // 循环计算 I1 矩阵
        for (int j = 0; j < n - 1; j++) {
            for (int k = 0; k < m - 1; k++) {
                I1(j, k) = (Ajk(j, k) + Ajk(j + 1, k) + Ajk(j + 1, k + 1) + Ajk(j, k + 1)) / 4 +
                    (Bjk(j, k) - Bjk(j + 1, k) - Bjk(j + 1, k + 1) + Bjk(j, k + 1)) / 16.0 +
                    (Cjk(j, k) + Cjk(j + 1, k) - Cjk(j + 1, k + 1) - Cjk(j, k + 1)) / 16.0 +
                    (Djk(j, k) - Djk(j + 1, k) + Djk(j + 1, k + 1) - Djk(j, k + 1)) / 64.0 +
                    (Ejk(j, k) + Ejk(j + 1, k) + Ejk(j + 1, k + 1) + Ejk(j, k + 1)) / 48.0 +
                    (Fjk(j, k) + Fjk(j + 1, k) + Fjk(j + 1, k + 1) + Fjk(j, k + 1)) / 48.0;
            }
        }
        I1(n - 1, span(0, m - 2)) = I1(0, span(0, m - 2));
        I1(span(0, n - 2), m - 1) = I1(span(0, n - 2), 0);
        I1(n - 1, m - 1) = I1(0, 0);
        break;
    }
    case 1: {
        for (int j = 1; j < n; ++j)
        {
            for (int k = 1; k < m; ++k)
            {
                I1(j, k) = (Ajk(j - 1, k - 1) + Ajk(j, k - 1) + Ajk(j, k) + Ajk(j - 1, k)) / 4.0 +
                    (Bjk(j - 1, k - 1) - Bjk(j, k - 1) - Bjk(j, k) + Bjk(j - 1, k)) / 16.0 +
                    (Cjk(j - 1, k - 1) + Cjk(j, k - 1) - Cjk(j, k) - Cjk(j - 1, k)) / 16.0 +
                    (Djk(j - 1, k - 1) - Djk(j, k - 1) + Djk(j, k) - Djk(j - 1, k)) / 64.0 +
                    (Ejk(j - 1, k - 1) + Ejk(j, k - 1) + Ejk(j, k) + Ejk(j - 1, k)) / 48.0 +
                    (Fjk(j - 1, k - 1) + Fjk(j, k - 1) + Fjk(j, k) + Fjk(j - 1, k)) / 48.0;
            }
        }
        I1(0, span(1, m - 1)) = I1(n - 1, span(1, m - 1));
        I1(span(1, n - 1), 0) = I1(span(1, n - 1), m - 1);
        I1(0, 0) = I1(n - 1, m - 1);
        break;
    }
    }
}

//测试通过 7*7矩阵
void weight(vec& omega1, vec& omega2, vec& omega3, const vec& u, int option) {
    //测试与matlab输出完全一致
    int n = u.n_elem;
    omega1.resize(n); omega2.resize(n); omega3.resize(n);
    vec ISj1(n, fill::zeros), ISj2(n, fill::zeros), ISj3(n, fill::zeros), alpha1(n), alpha2(n), alpha3(n);
    vec C(3);
    double xi = 1e-6;
    int p = 2;
    vec uj = u;

    // calculate ISj1
    for (int i = 2; i < n; i++) {
        ISj1(i) = 13.0 / 12.0 * pow(uj(i - 2) - 2.0 * uj(i - 1) + uj(i), 2) + 0.25 * pow(uj(i - 2) - 4.0 * uj(i - 1) + 3.0 * uj(i), 2);
    }
    ISj1(1) = 13.0 / 12.0 * pow(uj(n - 2) - 2.0 * uj(0) + uj(1), 2) + 0.25 * pow(uj(n - 2) - 4.0 * uj(0) + 3.0 * uj(1), 2);

    ISj1(0) = 13.0 / 12.0 * pow(uj(n - 3) - 2 * uj(n - 2) + uj(0), 2) + 1.0 / 4.0 * pow(uj(n - 3) - 4 * uj(n - 2) + 3 * uj(0), 2);

    // calculate ISj2
    for (int i = 1; i < n - 1; i++) {
        ISj2(i) = 13.0 / 12.0 * pow(uj(i - 1) - 2.0 * uj(i) + uj(i + 1), 2) + 0.25 * pow(uj(i - 1) - uj(i + 1), 2);
    }
    ISj2(0) = 13.0 / 12.0 * pow(uj(n - 2) - 2.0 * uj(0) + uj(1), 2) + 0.25 * pow(uj(n - 2) - uj(1), 2);
    ISj2(n - 1) = ISj2(0);

    // calculate ISj3

    for (int i = 0; i < n - 2; i++) {
        ISj3(i) = 13.0 / 12.0 * pow(u(i) - 2.0 * u(i + 1) + u(i + 2), 2) + 1.0 / 4.0 * pow(3.0 * u(i) - 4.0 * u(i + 1) + u(i + 2), 2);
    }
    ISj3(n - 2) = 13.0 / 12.0 * pow(u(n - 2) - 2.0 * u(n - 1) + u(1), 2) + 1.0 / 4.0 * pow(3.0 * u(n - 2) - 4.0 * u(n - 1) + u(1), 2);
    ISj3(n - 1) = 13.0 / 12.0 * pow(u(n - 1) - 2.0 * u(1) + u(2), 2) + 1.0 / 4.0 * pow(3.0 * u(n - 1) - 4.0 * u(1) + u(2), 2);


    if (option == 1) {
        C(0) = 3.0 / 16.0;
        C(1) = 5.0 / 8.0;
        C(2) = 3.0 / 16.0;
    }
    if (option == 2) {
        C(0) = 1.0 / 6.0;
        C(1) = 2.0 / 3.0;
        C(2) = 1.0 / 6.0;
    }


    for (int i = 0; i < n; i++) {
        alpha1(i) = C(0) / pow(xi + ISj1(i), p);
        alpha2(i) = C(1) / pow(xi + ISj2(i), p);
        alpha3(i) = C(2) / pow(xi + ISj3(i), p);
        double talpha = alpha1(i) + alpha2(i) + alpha3(i);
        omega1(i) = alpha1(i) / talpha;
        omega2(i) = alpha2(i) / talpha;
        omega3(i) = alpha3(i) / talpha;
    }

    //omega1.save("ISj1.csv", csv_ascii);
    //omega2.save("ISj2.csv", csv_ascii);
    //omega3.save("ISj3.csv", csv_ascii);
}






// 存在问题 
mat derivf(mat u, double hx, double hy) {

    int n = u.n_rows;
    int m = u.n_cols;
    mat f = u;
    mat dfj = zeros<mat>(n, m);
    mat ddfj = zeros<mat>(n, m);
    mat df = zeros<mat>(n, m);
    for (int j = 0; j < m; j++) {
        for (int i = 1; i < n - 1; i++) {
            dfj(i, j) = (f(i + 1, j) - f(i - 1, j)) / (2 * hx);
            ddfj(i, j) = (f(i + 1, j) - 2 * f(i, j) + f(i - 1, j)) / pow(hx, 2);
        }
        dfj(0, j) = (f(1, j) - f(n - 2, j)) / 2 / hx;
        dfj(n - 1, j) = dfj(0, j);
        ddfj(0, j) = (f(1, j) - 2 * f(0, j) + f(n - 2, j)) / pow(hx, 2);
        ddfj(n - 1, j) = ddfj(0, j);
        vec omega1(n), omega2(n), omega3(n);
        weight(omega1, omega2, omega3, f.col(j), 2);
        for (int i = 1; i < n - 1; i++) {
            df(i, j) = omega1(i) * (dfj(i - 1, j) + hx * ddfj(i - 1, j)) + omega2(i) * dfj(i, j)
                + omega3(i) * (dfj(i + 1, j) - hx * ddfj(i + 1, j));
        }
        df(0, j) = omega1(0) * (dfj(n - 2, j) + hx * ddfj(n - 2, j)) + omega2(0) * dfj(0, j)
            + omega3(0) * (dfj(1, j) - hx * ddfj(1, j));
        df(n - 1, j) = df(0, j);
    }

    mat g = u;
    mat dfg(n, m, fill::zeros);
    mat ddfg(n, m, fill::zeros);
    mat dg(n, m, fill::zeros);
    mat ddfg_temp(n, m, fill::zeros);
    vec omega4(m, fill::zeros), omega5(m, fill::zeros), omega6(m, fill::zeros);

    // calculate dfg and ddfg
    for (int i = 0; i < n; i++) {
        for (int j = 1; j < m - 1; j++) {
            dfg(i, j) = (g(i, j + 1) - g(i, j - 1)) / (2 * hy);
            ddfg_temp(i, j) = (g(i, j + 1) - 2 * g(i, j) + g(i, j - 1)) / (hy * hy);
        }
        dfg(i, 0) = (g(i, 1) - g(i, m - 2)) / (2 * hy);
        dfg(i, m - 1) = dfg(i, 0);
        ddfg_temp(i, 0) = (g(i, 1) - 2 * g(i, 0) + g(i, m - 2)) / (hy * hy);
        ddfg_temp(i, m - 1) = ddfg_temp(i, 0);

        weight(omega4, omega5, omega6, g.row(i).t(), 2);

        for (int j = 1; j < m - 1; j++) {
            dg(i, j) = omega4(j) * (dfg(i, j - 1) + hy * ddfg_temp(i, j - 1)) +
                omega5(j) * dfg(i, j) +
                omega6(j) * (dfg(i, j + 1) - hy * ddfg_temp(i, j + 1));
        }
        dg(i, 0) = omega4(0) * (dfg(i, m - 2) + hy * ddfg_temp(i, m - 2)) +
            omega5(0) * dfg(i, 0) +
            omega6(0) * (dfg(i, 1) - hy * ddfg_temp(i, 1));
        dg(i, m - 1) = dg(i, 0);
    }
    ////df有误(0,:)
    //df.save("df.csv", csv_ascii);
    ////有误 (:,0),(:,n-1)
    //dg.save("dg.csv", csv_ascii);
    dfg = -df - dg;
    return dfg;
}
void RK_NCE(mat& un, mat& uh, mat& uf, mat u, double hx, double hy, double dt, int value)
{
    if (value == 2)
    {
        mat K1 = derivf(u, hx, hy);
        mat K2 = derivf(u + dt * K1, hx, hy);
        un = u;
        uh = u + dt * (3.0 / 8.0 * K1 + 1.0 / 8.0 * K2);
        uf = u + dt * (1.0 / 2.0 * K1 + 1.0 / 2.0 * K2);
    }
    else if (value == 4)
    {
        mat K1 = derivf(u, hx, hy);
        mat K2 = derivf(u + 0.5 * dt * K1, hx, hy);
        mat K3 = derivf(u + 0.5 * dt * K2, hx, hy);
        mat K4 = derivf(u + dt * K3, hx, hy);
        un = u;
        uh = u + dt * (5.0 / 24.0 * K1 + 1.0 / 6.0 * K2 + 1.0 / 6.0 * K3 - 1.0 / 24.0 * K4);
        uf = u + dt * (1.0 / 6.0 * K1 + 1.0 / 3.0 * K2 + 1.0 / 3.0 * K3 + 1.0 / 6.0 * K4);
    }
}
//使用7*7的矩阵测试，e-17次
mat I2_forward(mat fun, mat fuh, mat fuf, mat gun, mat guh, mat guf, double hx, double hy, double dt) {

    int n = fun.n_rows;
    int m = fun.n_cols;

    mat I2_fx(n, m, fill::zeros);
    mat I2_gy(n, m, fill::zeros);
    for (int i = 0; i < n - 1; i++) {
        for (int j = 1; j < m - 2; j++) {
            I2_fx(i, j) = -dt / hx * (1.0 / 6 * (1.0 / 24 * (-fun(i + 1, j + 2) + 13 * fun(i + 1, j + 1) + 13 * fun(i + 1, j) - fun(i + 1, j - 1)) -
                1.0 / 24 * (-fun(i, j + 2) + 13 * fun(i, j + 1) + 13 * fun(i, j) - fun(i, j - 1)))
                + 4.0 / 6 * (1.0 / 24 * (-fuh(i + 1, j + 2) + 13 * fuh(i + 1, j + 1) + 13 * fuh(i + 1, j) - fuh(i + 1, j - 1)) -
                    1.0 / 24 * (-fuh(i, j + 2) + 13 * fuh(i, j + 1) + 13 * fuh(i, j) - fuh(i, j - 1)))
                + 1.0 / 6 * (1.0 / 24 * (-fuf(i + 1, j + 2) + 13 * fuf(i + 1, j + 1) + 13 * fuf(i + 1, j) - fuf(i + 1, j - 1)) -
                    1.0 / 24 * (-fuf(i, j + 2) + 13 * fuf(i, j + 1) + 13 * fuf(i, j) - fuf(i, j - 1))));
        }
    }
    //cout << "第一次：" << endl << I2_fx << endl;
    /*I2_fx(n,2:m-2)=I2_fx(1,2:m-2);*/
    I2_fx(n - 1, span(1, m - 3)) = I2_fx(0, span(1, m - 3));
    //cout << "第二次：" << endl << I2_fx << endl;

    for (int i = 0; i < n - 1; i++)
    {
        I2_fx(i, m - 1 - 1) = -dt / hx * (1.0 / 6 * (1.0 / 24 * (-fun(i + 1, 2) + 13 * fun(i + 1, m - 1) + 13 * fun(i + 1, m - 1 - 1) - fun(i + 1, m - 1 - 2)) -
            1.0 / 24 * (-fun(i, 2) + 13 * fun(i, m - 1) + 13 * fun(i, m - 1 - 1) - fun(i, m - 1 - 2)))
            + 4.0 / 6 * (1.0 / 24 * (-fuh(i + 1, 2) + 13 * fuh(i + 1, m - 1) + 13 * fuh(i + 1, m - 1 - 1) - fuh(i + 1, m - 1 - 2)) -
                1.0 / 24 * (-fuh(i, 2) + 13 * fuh(i, m - 1) + 13 * fuh(i, m - 1 - 1) - fuh(i, m - 1 - 2)))
            + 1.0 / 6 * (1.0 / 24 * (-fuf(i + 1, 2) + 13 * fuf(i + 1, m - 1) + 13 * fuf(i + 1, m - 1 - 1) - fuf(i + 1, m - 1 - 2)) -
                1.0 / 24 * (-fuf(i, 2) + 13 * fuf(i, m - 1) + 13 * fuf(i, m - 1 - 1) - fuf(i, m - 1 - 2))));
    }
    /*I2_fx(n,m-1)=I2_fx(1,m-1);*/
    I2_fx(n - 1, m - 2) = I2_fx(0, m - 2);


    for (int i = 0; i < n - 1; i++)
    {
        I2_fx(i, m - 1) = -dt / hx * (1.0 / 6 * (1.0 / 24 * (-fun(i + 1, 3) + 13 * fun(i + 1, 2) + 13 * fun(i + 1, 1) - fun(i + 1, m - 1 - 1)) -
            1.0 / 24 * (-fun(i, 3) + 13 * fun(i, 2) + 13 * fun(i, 1) - fun(i, m - 1 - 1)))
            + 4.0 / 6 * (1.0 / 24 * (-fuh(i + 1, 3) + 13 * fuh(i + 1, 2) + 13 * fuh(i + 1, 1) - fuh(i + 1, m - 1 - 1)) -
                1.0 / 24 * (-fuh(i, 3) + 13 * fuh(i, 2) + 13 * fuh(i, 1) - fuh(i, m - 1 - 1)))
            + 1.0 / 6 * (1.0 / 24 * (-fuf(i + 1, 3) + 13 * fuf(i + 1, 2) + 13 * fuf(i + 1, 1) - fuf(i + 1, m - 1 - 1)) -
                1.0 / 24 * (-fuf(i, 3) + 13 * fuf(i, 2) + 13 * fuf(i, 1) - fuf(i, m - 1 - 1))));
    }

    /*I2_fx(n,m)=I2_fx(1,m);
    I2_fx(1:n,1)=I2_fx(1:n,m);*/
    I2_fx(n - 1, m - 1) = I2_fx(0, m - 1);
    I2_fx.col(0) = I2_fx.col(m - 1);



    for (int i = 1; i < n - 2; i++) {
        for (int j = 0; j < m - 1; j++) {
            I2_gy(i, j) = -dt / hy * (
                1.0 / 6.0 * (
                    1.0 / 24.0 * (-gun(i + 2, j + 1) + 13 * gun(i + 1, j + 1) + 13 * gun(i, j + 1) -
                        gun(i - 1, j + 1)) -
                    1.0 / 24.0 * (-gun(i + 2, j) + 13 * gun(i + 1, j) + 13 * gun(i, j) - gun(i - 1, j))
                    ) +
                4.0 / 6.0 * (
                    1.0 / 24.0 * (-guh(i + 2, j + 1) + 13 * guh(i + 1, j + 1) + 13 * guh(i, j + 1) -
                        guh(i - 1, j + 1)) -
                    1.0 / 24.0 * (-guh(i + 2, j) + 13 * guh(i + 1, j) + 13 * guh(i, j) - guh(i - 1, j))
                    ) +
                1.0 / 6.0 * (
                    1.0 / 24.0 * (-guf(i + 2, j + 1) + 13 * guf(i + 1, j + 1) + 13 * guf(i, j + 1) -
                        guf(i - 1, j + 1)) -
                    1.0 / 24.0 * (-guf(i + 2, j) + 13 * guf(i + 1, j) + 13 * guf(i, j) - guf(i - 1, j))
                    )
                );
        }
    }
    I2_gy.submat(1, m - 1, n - 3, m - 1) = I2_gy.submat(1, 0, n - 3, 0);


    for (int j = 0; j < m - 1; j++) {
        I2_gy(n - 2, j) = -dt / hy * (
            1.0 / 6.0 * (1.0 / 24.0 * (-gun(1, j + 1) + 13 * gun(0, j + 1) + 13 * gun(n - 2, j + 1) - gun(n - 3, j + 1)) -
                1.0 / 24.0 * (-gun(1, j) + 13 * gun(0, j) + 13 * gun(n - 2, j) - gun(n - 3, j))) +
            4.0 / 6.0 * (1.0 / 24.0 * (-guh(1, j + 1) + 13 * guh(0, j + 1) + 13 * guh(n - 2, j + 1) - guh(n - 3, j + 1)) -
                1.0 / 24.0 * (-guh(1, j) + 13 * guh(0, j) + 13 * guh(n - 2, j) - guh(n - 3, j))) +
            1.0 / 6.0 * (1.0 / 24.0 * (-guf(1, j + 1) + 13 * guf(0, j + 1) + 13 * guf(n - 2, j + 1) - guf(n - 3, j + 1)) -
                1.0 / 24.0 * (-guf(1, j) + 13 * guf(0, j) + 13 * guf(n - 2, j) - guf(n - 3, j))));

    }
    I2_gy(n - 2, m - 1) = I2_gy(n - 2, 0);
    for (int j = 0; j < m - 1; j++)
    {
        I2_gy(n - 1, j) = -dt / hy * (1.0 / 6.0 * (1.0 / 24.0 * (-gun(2, j + 1) + 13.0 * gun(1, j + 1) + 13.0 * gun(0, j + 1) - gun(n - 2, j + 1))
            - 1.0 / 24.0 * (-gun(2, j) + 13.0 * gun(1, j) + 13.0 * gun(0, j) - gun(n - 2, j)))
            + 4.0 / 6.0 * (1.0 / 24.0 * (-guh(2, j + 1) + 13.0 * guh(1, j + 1) + 13.0 * guh(0, j + 1) - guh(n - 2, j + 1))
                - 1.0 / 24.0 * (-guh(2, j) + 13.0 * guh(1, j) + 13.0 * guh(0, j) - guh(n - 2, j)))
            + 1.0 / 6.0 * (1.0 / 24.0 * (-guf(2, j + 1) + 13.0 * guf(1, j + 1) + 13.0 * guf(0, j + 1) - guf(n - 2, j + 1))
                - 1.0 / 24.0 * (-guf(2, j) + 13.0 * guf(1, j) + 13.0 * guf(0, j) - guf(n - 2, j))));

    }

    I2_gy(n - 1, m - 1) = I2_gy(n - 1, 0);
    I2_gy.row(0) = I2_gy.row(n - 1);

    mat I2 = I2_fx + I2_gy;

    return I2;
}
//测试通过
mat I2_backward(mat fun, mat fuh, mat fuf, mat gun, mat guh, mat guf, double hx, double hy, double dt) {

    int n = fun.n_rows;
    int m = fun.n_cols;
    mat I2_fx(n, m, fill::zeros);
    mat I2_gy(n, m, fill::zeros);

    // I2_fx
    for (int i = 1; i < n; ++i) {
        for (int j = 2; j < m - 1; ++j) {
            I2_fx(i, j) = -dt / hx * (1.0 / 6 * (1.0 / 24 * (-fun(i, j + 1) + 13 * fun(i, j) + 13 * fun(i, j - 1) - fun(i, j - 2)) -
                1.0 / 24 * (-fun(i - 1, j + 1) + 13 * fun(i - 1, j) + 13 * fun(i - 1, j - 1) - fun(i - 1, j - 2)))
                + 4.0 / 6 * (1.0 / 24 * (-fuh(i, j + 1) + 13 * fuh(i, j) + 13 * fuh(i, j - 1) - fuh(i, j - 2)) -
                    1.0 / 24 * (-fuh(i - 1, j + 1) + 13 * fuh(i - 1, j) + 13 * fuh(i - 1, j - 1) - fuh(i - 1, j - 2)))
                + 1.0 / 6 * (1.0 / 24 * (-fuf(i, j + 1) + 13 * fuf(i, j) + 13 * fuf(i, j - 1) - fuf(i, j - 2)) -
                    1.0 / 24 * (-fuf(i - 1, j + 1) + 13 * fuf(i - 1, j) + 13 * fuf(i - 1, j - 1) - fuf(i - 1, j - 2))));
        }
    }

    I2_fx(0, span(2, m - 2)) = I2_fx(n - 1, span(2, m - 2));

    for (int i = 1; i < n; ++i) {
        I2_fx(i, 1) = -dt / hx * (1.0 / 6 * (1.0 / 24 * (-fun(i, 3) + 13 * fun(i, 2) + 13 * fun(i, 1) - fun(i, m - 1 - 1)) -
            1.0 / 24 * (-fun(i - 1, 3) + 13 * fun(i - 1, 2) + 13 * fun(i - 1, 1) - fun(i - 1, m - 1 - 1)))
            + 4.0 / 6 * (1.0 / 24 * (-fuh(i, 3) + 13 * fuh(i, 2) + 13 * fuh(i, 1) - fuh(i, m - 1 - 1)) -
                1.0 / 24 * (-fuh(i - 1, 3) + 13 * fuh(i - 1, 2) + 13 * fuh(i - 1, 1) - fuh(i - 1, m - 1 - 1)))
            + 1.0 / 6 * (1.0 / 24 * (-fuf(i, 3) + 13 * fuf(i, 2) + 13 * fuf(i, 1) - fuf(i, m - 1 - 1)) -
                1.0 / 24 * (-fuf(i - 1, 3) + 13 * fuf(i - 1, 2) + 13 * fuf(i - 1, 1) - fuf(i - 1, m - 1 - 1))));
    }
    I2_fx(0, 1) = I2_fx(n - 1, 1);


    for (int i = 1; i < n; i++) {
        I2_fx(i, 0) = -dt / hx * (1.0 / 6 * (1.0 / 24 * (-fun(i, 2) + 13 * fun(i, 1) + 13 * fun(i, m - 1 - 1) - fun(i, m - 1 - 2)) -
            1.0 / 24 * (-fun(i - 1, 2) + 13 * fun(i - 1, 1) + 13 * fun(i - 1, m - 1 - 1) - fun(i - 1, m - 1 - 2)))
            + 4.0 / 6 * (1.0 / 24 * (-fuh(i, 2) + 13 * fuh(i, 1) + 13 * fuh(i, m - 1 - 1) - fuh(i, m - 1 - 2)) -
                1.0 / 24 * (-fuh(i - 1, 2) + 13 * fuh(i - 1, 1) + 13 * fuh(i - 1, m - 1 - 1) - fuh(i - 1, m - 1 - 2)))
            + 1.0 / 6 * (1.0 / 24 * (-fuf(i, 2) + 13 * fuf(i, 1) + 13 * fuf(i, m - 1 - 1) - fuf(i, m - 1 - 2)) -
                1.0 / 24 * (-fuf(i - 1, 2) + 13 * fuf(i - 1, 1) + 13 * fuf(i - 1, m - 1 - 1) - fuf(i - 1, m - 1 - 2))));
    }
    I2_fx(0, 0) = I2_fx(n - 1, 0);
    I2_fx.col(m - 1) = I2_fx.col(0);



    for (int i = 2; i < n - 1; i++) {
        for (int j = 1; j < m; j++) {
            I2_gy(i, j) = -dt / hy * (1.0 / 6 * (1.0 / 24 * (-gun(i + 1, j) + 13 * gun(i, j) + 13 * gun(i - 1, j) - gun(i - 2, j))
                - 1.0 / 24 * (-gun(i + 1, j - 1) + 13 * gun(i, j - 1) + 13 * gun(i - 1, j - 1) - gun(i - 2, j - 1)))
                + 4.0 / 6 * (1.0 / 24 * (-guh(i + 1, j) + 13 * guh(i, j) + 13 * guh(i - 1, j) - guh(i - 2, j))
                    - 1.0 / 24 * (-guh(i + 1, j - 1) + 13 * guh(i, j - 1) + 13 * guh(i - 1, j - 1) - guh(i - 2, j - 1)))
                + 1.0 / 6 * (1.0 / 24 * (-guf(i + 1, j) + 13 * guf(i, j) + 13 * guf(i - 1, j) - guf(i - 2, j))
                    - 1.0 / 24 * (-guf(i + 1, j - 1) + 13 * guf(i, j - 1) + 13 * guf(i - 1, j - 1) - guf(i - 2, j - 1))));
        }
    }
    I2_gy(span(2, n - 2), 0) = I2_gy(span(2, n - 2), m - 1);
    for (int j = 1; j < m; j++)
    {
        I2_gy(1, j) = -dt / hy * (1.0 / 6 * (1.0 / 24 * (-gun(3, j) + 13 * gun(2, j) + 13 * gun(1, j) - gun(n - 1 - 1, j)) -
            1.0 / 24 * (-gun(3, j - 1) + 13 * gun(2, j - 1) + 13 * gun(1, j - 1) - gun(n - 1 - 1, j - 1)))
            + 4.0 / 6 * (1.0 / 24 * (-guh(3, j) + 13 * guh(2, j) + 13 * guh(1, j) - guh(n - 1 - 1, j)) -
                1.0 / 24 * (-guh(3, j - 1) + 13 * guh(2, j - 1) + 13 * guh(1, j - 1) - guh(n - 1 - 1, j - 1)))
            + 1.0 / 6 * (1.0 / 24 * (-guf(3, j) + 13 * guf(2, j) + 13 * guf(1, j) - guf(n - 1 - 1, j)) -
                1.0 / 24 * (-guf(3, j - 1) + 13 * guf(2, j - 1) + 13 * guf(1, j - 1) - guf(n - 1 - 1, j - 1))));

    }
    I2_gy(1, 0) = I2_gy(1, m - 1);


    for (int j = 1; j < m; j++)
    {
        I2_gy(0, j) = -dt / hy * (1.0 / 6 * (1.0 / 24 * (-gun(2, j) + 13 * gun(1, j) + 13 * gun(n - 1 - 1, j) - gun(n - 1 - 2, j)) -
            1.0 / 24 * (-gun(2, j - 1) + 13 * gun(1, j - 1) + 13 * gun(n - 1 - 1, j - 1) - gun(n - 1 - 2, j - 1)))
            + 4.0 / 6 * (1.0 / 24 * (-guh(2, j) + 13 * guh(1, j) + 13 * guh(n - 1 - 1, j) - guh(n - 1 - 2, j)) -
                1.0 / 24 * (-guh(2, j - 1) + 13 * guh(1, j - 1) + 13 * guh(n - 1 - 1, j - 1) - guh(n - 1 - 2, j - 1)))
            + 1.0 / 6 * (1.0 / 24 * (-guf(2, j) + 13 * guf(1, j) + 13 * guf(n - 1 - 1, j) - guf(n - 1 - 2, j)) -
                1.0 / 24 * (-guf(2, j - 1) + 13 * guf(1, j - 1) + 13 * guf(n - 1 - 1, j - 1) - guf(n - 1 - 2, j - 1))));
    }
    I2_gy(0, 0) = I2_gy(0, m - 1);
    I2_gy.row(n - 1) = I2_gy.row(0);
    mat I2 = I2_fx + I2_gy;
    return I2;
}

mat function_f(mat u) {
    return u;
}

mat function_g(mat u) {
    return u;
}

inline bool save_npy(const mat& x, const std::string& filename)
{
    std::ofstream f(filename.c_str(), std::fstream::binary);

    bool save_okay = f.is_open();

    if (save_okay)
    {   
        //存放头文件
        char part1[0xA] = { 0x93,0x4E,0X55,0X4D,0X50,0X59,0X01,0X00,0X76,0X00 };
        char part2[118]= "{'descr': '<f8', 'fortran_order': False, 'shape': (80,), }";
        /*
        memset(part2, 0x20, sizeof(part2));*/
        part2[117] = 0x0A;
        f.write(part1, sizeof(part1));
        f.write(part2,sizeof(part2));/*
        f << x.n_rows << ' ' << x.n_cols << '\n';*/
        f.write(reinterpret_cast<const char*>(x.mem), std::streamsize(x.n_elem * sizeof(double)));
        f.flush();
        f.close();

    }

    return f.good();
}



int main()
{
    int N = 80;
    int M = 80;
    double lambda = 0.425;
    double hx = 1.0 / (N - 1);
    double hy = 1.0 / (M - 1);
    double dt = lambda * hx;
    int tn = round(1 / dt);

    mat x = linspace<mat>(0, 1, N);
    mat y = linspace<mat>(0, 1, M);



    vec t = linspace<vec>(0, tn * dt, tn + 1);

    vec x1 = x + hx / 2;
    vec y1 = y + hy / 2;
    //理论值
    cube u2(N, M, tn + 1);
    for (int i = 0; i <= tn; i++)
    {
        int option = i % 2;
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < M; k++)
            {
                switch (option)
                {
                case 1:
                    u2(j, k, i) = pow(sin(datum::pi * (x(j) - t(i))), 2) * pow(sin(datum::pi * (y(k) - t(i))), 2);
                    break;
                case 0:
                    u2(j, k, i) = pow(sin(datum::pi * (x1(j) - t(i))), 2) * pow(sin(datum::pi * (y1(k) - t(i))), 2);
                    break;
                }
            }
        }
    }
    //模拟值
    cube u(N, M, tn, fill::zeros);

    // 初始化 t=0
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            u(i, j, 0) = pow(sin(datum::pi * i * hx), 2) * pow(sin(datum::pi * j * hy), 2); // t=0
        }
    }
    //u.slice(0).save("t0.csv", csv_ascii);
    // 进行时间迭代
    for (int i = 1; i < tn; i++) {
        int option = (i+1) % 2;

        mat I1, Ajk, omegaNE, omegaNW, omegaSE, omegaSW, omegaC;
        //存在问题,经排查解决 3.2 13.30 未解决3.3
        Rjx2(u.slice(i - 1), option, I1, Ajk, omegaNE, omegaNW, omegaSE, omegaSW, omegaC);
        //Ajk.save("Ajk.csv", csv_ascii);
        mat un, uh, uf;
        //un无误  uh,uf有问题,问题来自Ajk
        RK_NCE(un, uh, uf, Ajk, hx, hy, dt, 4); // Ajk or u.slice(i-1)
     

        mat fun = function_f(un);
        mat fuh = function_f(uh);
        mat fuf = function_f(uf);
        mat gun = function_g(un);
        mat guh = function_g(uh);
        mat guf = function_g(uf);
        mat I2;
        //vector<mat> matlist = { un,uh,uf };
        //for (i = 0; i < matlist.size(); i++) {
        //    matlist[0].save(to_string(i) + ".csv", csv_ascii);
        //}
        switch (option)
        {
        case 0:
            I2 = I2_forward(fun, fuh, fuf, gun, guh, guf, hx, hy, dt);
            break;
        case 1:
            I2 = I2_backward(fun, fuh, fuf, gun, guh, guf, hx, hy, dt);
            break;
        }
        //I2,在0，n-2,n-1存在很大差异
        mat I0 = I1 + I2;
        //I1.save("I1.csv", csv_ascii);
        //I2.save("I2.csv", csv_ascii);
        //I0.save("I0.csv", csv_ascii);
        u.slice(i) = I0;

        //// 输出进度
        //cout << "time step: " << i << endl;
    }


    //准备使用python实现数据可视化

    //遇到了麻烦，arma列优先，numpy行优先
    save_npy(x.t(), "xtest.npy");
    u.save("uraw", raw_binary);
    x.save("x");
    y.save("y");
    x1.save("x1");
    y1.save("y1");
    t.save("t");
    u2.save("u2");

    return 0;
}





