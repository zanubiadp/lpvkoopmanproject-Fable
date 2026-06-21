% Cross-check driver executed by GNU Octave (or MATLAB).
%
% Loads the test problem written by run_crosscheck.py, evaluates the thesis
% reference implementation of the eigenvalue-learning core, and writes the
% results back as CSV for comparison against the lpvkoopman implementation:
%
%   * getCostGradientKordacc_re_fast.m  - real conjugate-pair cost/gradient
%     (thesis Appendix A formulation, continuous time), plus the boundary
%     least-squares solution and the fitted trajectory values;
%   * eigOptim_grad.m                   - complex discrete-time cost
%     (powers of mu = exp(lambda*Ts)), as used by build_A.m.
%
% Expects to be run from the directory containing the input CSVs, with the
% thesis repo path passed via the environment variable MATLAB_REPO.

repo = getenv('MATLAB_REPO');
addpath(fullfile(repo, 'StateSpaceMatricesBuilding', 'build_A_package', 'Functions'));

x    = csvread('in_theta.csv');    % [Re1 Im1 Re2 Im2 ... | reals]
h    = csvread('in_h.csv');        % (Ms+1)*Mt, trajectory-major
t    = csvread('in_t.csv');        % Ms+1 shared sample times
meta = csvread('in_meta.csv');     % [n_cc, Mt, Ts]
n_cc = meta(1); Mt = meta(2); Ts = meta(3);

% --- thesis Appendix-A real conjugate-pair formulation ------------------
[J, grad, L] = getCostGradientKordacc_re_fast(x(:), h(:), t(:), n_cc);
q    = (L' * L) \ (L' * h(:));
hhat = L * q;

csvwrite('out_J_real.csv', J);
csvwrite('out_grad_real.csv', grad(:));
csvwrite('out_hhat_real.csv', full(hhat(:)));

% Finite-difference gradient of the reference cost itself (arbitrates
% between the two analytic gradients).
grad_fd = zeros(length(x), 1);
step = 1e-4;
for k = 1:length(x)
    e = zeros(size(x(:))); e(k) = step;
    Jp = getCostGradientKordacc_re_fast(x(:) + e, h(:), t(:), n_cc);
    Jm = getCostGradientKordacc_re_fast(x(:) - e, h(:), t(:), n_cc);
    grad_fd(k) = (Jp - Jm) / (2 * step);
end
csvwrite('out_grad_fd.csv', grad_fd);

% --- discrete-time complex formulation (build_A.m path) -----------------
lams = [];
for ii = 1:2:2*n_cc
    lams = [lams; x(ii) + 1i * x(ii + 1); x(ii) - 1i * x(ii + 1)];
end
for ii = 2*n_cc+1:length(x)
    lams = [lams; x(ii)];
end
mu = exp(Ts * lams);
xc = [real(mu); imag(mu)];
TrajLen = length(t);
[J2, grad2] = eigOptim_grad(xc, h(:), Mt, TrajLen);
csvwrite('out_J_discrete.csv', J2);

fprintf('octave: J_real = %.12e, J_discrete = %.12e\n', J, J2);
