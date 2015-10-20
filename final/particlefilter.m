clear all;
clc;
echo off;
path('./ukf',path);

% INITIALISATION AND PARAMETERS:
% ==============================

no_of_runs = 100;            % number of experiments to generate statistical
                            % averages
doPlot = 0;                 % 1 plot online. 0 = only plot at the end.
sigma =  1e-4;              % Variance of the Gaussian measurement noise.
g1 = 3;                     % Paramater of Gamma transition prior.
g2 = 2;                     % Parameter of Gamman transition prior.
                            % Thus mean = 3/2 and var = 3/4.
T = 60;                     % Number of time steps.
R = 1e-4;                   % EKF's measurement noise variance. 
Q = 3/4;                    % EKF's process noise variance.
P0 = 3/4;                   % EKF's initial variance of the states.

N = 1000;                     % Number of particles.
M = 264;                      % Number of sensors.

Q_pfekf = 10*3/4;
R_pfekf = 1e-1;

Q_pfukf = 2*3/4;
R_pfukf = 1e-1;
          
alpha = 1;                  % UKF : point scaling parameter
beta  = 0;                  % UKF : scaling parameter for higher order terms of Taylor series expansion 
kappa = 2;                  % UKF : sigma point selection scaling parameter (best to leave this = 0)

d = 3;
xpf = zeros(T,1); 

% GENERATE THE DATA:
% ==================
mu = [0 0 0.01 0.01]';
Cu = diag([10,10,0.1,0.1],0);
x0 = mvnrnd(mu,Cu,1);
x = zeros(T,4);
y = zeros(T,M);
processNoise = zeros(T,1);
measureNoise = zeros(T,1);
x(1) = 1;                         % Initial state.
for t=2:T
  processNoise(t) = gengamma(g1,g2);  
  measureNoise(t) = sqrt(sigma)*randn(1,1);    
  x(t) = feval('ffun',x(t-1),t) +processNoise(t);     % Gamma transition prior.  
  y(t) = feval('hfun',x(t),t) + measureNoise(t);      % Gaussian likelihood.
end;  

% PLOT THE GENERATED DATA:
% ========================
figure(1)
clf;
plot(1:T,x,'r',1:T,y,'b');
ylabel('Data','fontsize',15);
xlabel('Time','fontsize',15);
legend('States (x)','Observations(y)');

%%%%%%%%%%%%%%%  PERFORM PARTICLE FILTER  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%  ==============================  %%%%%%%%%%%%%%%%%%%%%

% INITIALISATION:
% ==============
xparticle_pf = ones(T,N);        % These are the particles for the estimate
                                 % of x. Note that there's no need to store
                                 % them for all t. We're only doing this to
                                 % show you all the nice plots at the end.
xparticlePred_pf = ones(T,N);    % One-step-ahead predicted values of the states.
yPred_pf = ones(T,N);            % One-step-ahead predicted values of y.
w = ones(T,N);                   % Importance weights.

disp(' ');
 

fprintf('PF ');
fprintf('\n')
tic;                             % Initialize timer for benchmarking
xpf(1,:) = x(1); 
for t=2:T,  
     
  % PREDICTION STEP:
  % ================ 
  % We use the transition prior as proposal.
  for i=1:N,
    xparticlePred_pf(t,i) = feval('ffun',xparticle_pf(t-1,i),t) + gengamma(g1,g2);   
  end;

  % EVALUATE IMPORTANCE WEIGHTS:
  % ============================
  % For our choice of proposal, the importance weights are give by:  
  for i=1:N,
    yPred_pf(t,i) = feval('hfun',xparticlePred_pf(t,i),t);        
    lik = inv(sqrt(sigma)) * exp(-0.5*inv(sigma)*((y(t)-yPred_pf(t,i))^(2))) ...
	  + 1e-99; % Deal with ill-conditioning.
    w(t,i) = lik;    
  end;  
  w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.
  
  % SELECTION STEP:
  % ===============
  % Here, we give you the choice to try three different types of
  % resampling algorithms. Note that the code for these algorithms
  % applies to any problem!
  outIndex = systematicR(1:N,w(t,:)');      % Systematic resampling.
  xparticle_pf(t,:) = xparticlePred_pf(t,outIndex); % Keep particles with
                                                   % resampled indices.
  w(t,:) = 1/N ;
  xpf(t,1) = xparticle_pf(t,:)*w(t,:)';
end;   % End of t loop.

rmsError_pf = sqrt(inv(T)*sum((x'-xpf').^(2)));
disp(' ');
disp('Root mean square (RMS) errors');
disp('-----------------------------');
disp(' ');
disp(['PF     = ' num2str(rmsError_pf)]);