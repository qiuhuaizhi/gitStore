% PURPOSE : Demonstrate the differences between the following filters on the same problem:
%           
%           1) Extended Kalman Filter  (EKF)
%           2) Unscented Kalman Filter (UKF)
%           3) Particle Filter         (PF)
%           4) PF with EKF proposal    (PFEKF)
%           5) PF with UKF proposal    (PFUKF)


% For more details refer to:

% AUTHORS  : Nando de Freitas      (jfgf@cs.berkeley.edu)
%            Rudolph van der Merwe (rvdmerwe@ece.ogi.edu)
% DATE     : 17 August 2000

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

N = 100;                     % Number of particles.


Q_pfekf = 10*3/4;
R_pfekf = 1e-1;

Q_pfukf = 2*3/4;
R_pfukf = 1e-1;
			    
alpha = 1;                  % UKF : point scaling parameter
beta  = 0;                  % UKF : scaling parameter for higher order terms of Taylor series expansion 
kappa = 2;                  % UKF : sigma point selection scaling parameter (best to leave this = 0)

d = 3;

%**************************************************************************************

% SETUP BUFFERS TO STORE PERFORMANCE RESULTS
% ==========================================

rmsError_ekf      = zeros(1,no_of_runs);
rmsError_ukf      = zeros(1,no_of_runs);
rmsError_pf       = zeros(1,no_of_runs);
rmsError_pfMC     = zeros(1,no_of_runs);
rmsError_pfekf    = zeros(1,no_of_runs);
rmsError_pfekfMC  = zeros(1,no_of_runs);
rmsError_pfukf    = zeros(1,no_of_runs);
rmsError_pfukfMC  = zeros(1,no_of_runs);

time_pf       = zeros(1,no_of_runs);     
time_pfqmc     = zeros(1,no_of_runs);
time_pfekf    = zeros(1,no_of_runs);
time_pfqmcekf  = zeros(1,no_of_runs);
time_pfukf    = zeros(1,no_of_runs);
time_pfqmcukf  = zeros(1,no_of_runs);
time_sqmc    = zeros(1,no_of_runs);
time_sqmcukf  = zeros(1,no_of_runs);

mu_ekf     =  zeros(T,1);     
mu_ukf     = zeros(T,1);
xpf       = zeros(T,1);     
xpfqmc     = zeros(T,1);
xpfekf    = zeros(T,1);
xpfqmcekf  = zeros(T,1);
xpfukf    = zeros(T,1);
xpfqmcukf  = zeros(T,1);
xsqmc    = zeros(T,1);
xsqmcukf  = zeros(T,1);
%**************************************************************************************

% MAIN LOOP

for j=1:no_of_runs,

  rand('state',sum(100*clock));   % Shuffle the pack!
  randn('state',sum(100*clock));   % Shuffle the pack!
  

% GENERATE THE DATA:
% ==================

x = zeros(T,1);
y = zeros(T,1);
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
% figure(1)
% clf;
% plot(1:T,x,'r',1:T,y,'b');
% ylabel('Data','fontsize',15);
% xlabel('Time','fontsize',15);
% legend('States (x)','Observations(y)');

%%%%%%%%%%%%%%%  PERFORM EKF and UKF ESTIMATION  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%  ==============================  %%%%%%%%%%%%%%%%%%%%%

% INITIALISATION:
% ==============
mu_ekf = ones(T,1);     % EKF estimate of the mean of the states.
P_ekf = P0*ones(T,1);   % EKF estimate of the variance of the states.

mu_ukf = mu_ekf;        % UKF estimate of the mean of the states.
P_ukf = P_ekf;          % UKF estimate of the variance of the states.

yPred = ones(T,1);      % One-step-ahead predicted values of y.
mu_ekfPred = ones(T,1); % EKF O-s-a estimate of the mean of the states.
PPred = ones(T,1);      % EKF O-s-a estimate of the variance of the states.

disp(' ');
fprintf('EKF & UKF ');
fprintf('\n')
for t=2:T,    

  
  % PREDICTION STEP:
  % ================ 
  mu_ekfPred(t) = feval('ffun',mu_ekf(t-1),t);
  Jx = 0.5;                             % Jacobian for ffun.
  PPred(t) = Q + Jx*P_ekf(t-1)*Jx'; 
  
  % CORRECTION STEP:
  % ================
  yPred(t) = feval('hfun',mu_ekfPred(t),t);
  if t<=30,
    Jy = 2*0.2*mu_ekfPred(t);                 % Jacobian for hfun.
  else
    Jy = 0.5;
  %  Jy = cos(mu_ekfPred(t))/2;
  %   Jy = 2*mu_ekfPred(t)/4;                 % Jacobian for hfun. 
  end;
  M = R + Jy*PPred(t)*Jy';                 % Innovations covariance.
  K = PPred(t)*Jy'*inv(M);                 % Kalman gain.
  mu_ekf(t) = mu_ekfPred(t) + K*(y(t)-yPred(t));
  P_ekf(t) = PPred(t) - K*Jy*PPred(t);
  
  % Full Unscented Kalman Filter step
  % =================================
  [mu_ukf(t),P_ukf(t)]=ukf(mu_ukf(t-1),P_ukf(t-1),[],Q,'ukf_ffun',y(t),R,'ukf_hfun',t,alpha,beta,kappa);  
  
end;   % End of t loop.



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

time_pf(j) = toc;    % How long did this take?


%%%%%%%%%%%%%%  PERFORM QSIR  %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%  ========================================  %%%%%%%%%%%%%%%%

% INITIALISATION:
% ==============
xparticle_pfQMC = ones(T,N);      % These are the particles for the estimate
                                 % of x. Note that there's no need to store
                                 % them for all t. We're only doing this to
                                 % show you all the nice plots at the end.
xparticlePred_pfQMC = ones(T,N);  % One-step-ahead predicted values of the states.
yPred_pfQMC = ones(T,N);          % One-step-ahead predicted values of y.
w = ones(T,N);                   % Importance weights.
xpfqmc(1,:) = x(1) ;
disp(' '); 
                            % Initialize timer for benchmarking
fprintf('QSIR ');
fprintf('\n')
tic; 
for t=2:T,    

  
  % PREDICTION STEP:
  % ================ 
  % We use the transition prior as proposal.
  for i=1:N,
    xparticlePred_pfQMC(t,i) = feval('ffun',xparticle_pfQMC(t-1,i),t)+gengamma(g1,g2);    
  end;
  

    % EVALUATE IMPORTANCE WEIGHTS:
  % ============================
  % For our choice of proposal, the importance weights are give by:  
  for i=1:N,
    yPred_pfQMC(t,i) = feval('hfun',xparticlePred_pfQMC(t,i),t);        
    w(t,i) = inv(sqrt(sigma)) * exp(-0.5*inv(sigma)*((y(t)-yPred_pfQMC(t,i))^(2))) ...
	  + 1e-99; % Deal with ill-conditioning.   
  end;  
  w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.  
  xpfqmc(t,1)=xparticlePred_pfQMC(t,:)*(w(t,:))';     
     
  [xparticle_pfQMC(t,:),w(t,:)] = feval('qmcmultiresample',xparticlePred_pfQMC(t,:),w(t,:),N,d);  
  w(t,:) = w(t,:)./sum(w(t,:)); 
 
  
end;   % End of t loop.

time_pfqmc(j) = toc;    % How long did this take?


% %%%%%%%%%%%%%%%  PERFORM SEQUENTIAL MONTE CARLO  %%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%  ======== EKF proposal ========  %%%%%%%%%%%%%%%%%%%%%
% 
% % INITIALISATION:
% % ==============
% xparticle_pfekf = ones(T,N);        % These are the particles for the estimate
%                                     % of x. Note that there's no need to store
%                                     % them for all t. We're only doing this to
%                                     % show you all the nice plots at the end.
% Pparticle_pfekf = P0*ones(T,N);     % Particles for the covariance of x.
% xparticlePred_pfekf = ones(T,N);    % One-step-ahead predicted values of the states.
% PparticlePred_pfekf = ones(T,N);    % One-step-ahead predicted values of P.
% yPred_pfekf = ones(T,N);            % One-step-ahead predicted values of y.
% w = ones(T,N);                      % Importance weights.
% muPred_pfekf = ones(T,1);           % EKF O-s-a estimate of the mean of the states.
% PPred_pfekf = ones(T,1);            % EKF O-s-a estimate of the variance of the states.
% mu_pfekf = ones(T,1);               % EKF estimate of the mean of the states.
% P_pfekf = P0*ones(T,1);             % EKF estimate of the variance of the states.
% xpfekf(1,:) = x(1) ;
% disp(' ');
% 
% tic;                                % Initialize timer for benchmarking
%   fprintf('PF-EKF');
%   fprintf('\n')
% for t=2:T,    
% 
%   
%   % PREDICTION STEP:
%   % ================ 
%   % We use the EKF as proposal.
%   for i=1:N,
%     muPred_pfekf(t) = feval('ffun',xparticle_pfekf(t-1,i),t);
%     Jx = 0.5;                                 % Jacobian for ffun.
%     PPred_pfekf(t) = Q_pfekf + Jx*Pparticle_pfekf(t-1,i)*Jx'; 
%     yPredTmp = feval('hfun',muPred_pfekf(t),t);
%     if t<=30,
%       Jy = 2*0.2*muPred_pfekf(t);                     % Jacobian for hfun.
%     else
%       Jy = 0.5;
%     end;
%     M = R_pfekf + Jy*PPred_pfekf(t)*Jy';                  % Innovations covariance.
%     K = PPred_pfekf(t)*Jy'*inv(M);                  % Kalman gain.
%     mu_pfekf(t,i) = muPred_pfekf(t) + K*(y(t)-yPredTmp); % Mean of proposal.
%     P_pfekf(t) = PPred_pfekf(t) - K*Jy*PPred_pfekf(t);          % Variance of proposal.
%     xparticlePred_pfekf(t,i) = mu_pfekf(t,i) + sqrtm(P_pfekf(t))*randn(1,1);
%     PparticlePred_pfekf(t,i) = P_pfekf(t);
%   end;
% 
%   % EVALUATE IMPORTANCE WEIGHTS:
%   % ============================
%   % For our choice of proposal, the importance weights are give by:  
%   for i=1:N,
%     yPred_pfekf(t,i) = feval('hfun',xparticlePred_pfekf(t,i),t);        
%     lik = inv(sqrt(sigma)) * exp(-0.5*inv(sigma)*((y(t)-yPred_pfekf(t,i))^(2)))+1e-99;
%     prior = ((xparticlePred_pfekf(t,i)-xparticle_pfekf(t-1,i))^(g1-1)) ...
% 		 * exp(-g2*(xparticlePred_pfekf(t,i)-xparticle_pfekf(t-1,i)));
%     proposal = inv(sqrt(PparticlePred_pfekf(t,i))) * ...
% 	       exp(-0.5*inv(PparticlePred_pfekf(t,i)) *((xparticlePred_pfekf(t,i)-mu_pfekf(t,i))^(2)));
%     w(t,i) = lik*prior/proposal;      
%   end;  
%   w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.
%   
%   % SELECTION STEP:
%   % ===============
%   % Here, we give you the choice to try three different types of
%   % resampling algorithms. Note that the code for these algorithms
%   % applies to any problem!
%   outIndex = systematicR(1:N,w(t,:)');      % Systematic resampling.
% 
%   xparticle_pfekf(t,:) = xparticlePred_pfekf(t,outIndex); % Keep particles with % resampled indices.                                             
%   Pparticle_pfekf(t,:) = PparticlePred_pfekf(t,outIndex); 
%   w(t,:) = 1/N ;
%   xpfekf(t,1) = xparticle_pfekf(t,:)*w(t,:)';
% end;   % End of t loop.
% 
% time_pfekf(j) = toc;
% 
% 
% 
% %%%%%%%%%%%%%%  PERFORM QSIR  %%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%  EKF proposal  %%%%%%%%%%%%%%%%
% 
% % INITIALISATION:
% % ==============
% xparticle_pfekfQMC = ones(T,N);        % These are the particles for the estimate
%                                       % of x. Note that there's no need to store
%                                       % them for all t. We're only doing this to
%                                       % show you all the nice plots at the end.
% Pparticle_pfekfQMC = P0*ones(T,N);     % Particles for the covariance of x.
% xparticlePred_pfekfQMC = ones(T,N);    % One-step-ahead predicted values of the states.
% PparticlePred_pfekfQMC = ones(T,N);    % One-step-ahead predicted values of P.
% yPred_pfekfQMC = ones(T,N);            % One-step-ahead predicted values of y.
% w = ones(T,N);                        % Importance weights.
% muPred_pfekfQMC = ones(T,1);           % EKF O-s-a estimate of the mean of the states.
% PPred_pfekfQMC = ones(T,1);            % EKF O-s-a estimate of the variance of the states.
% mu_pfekfQMC = ones(T,1);               % EKF estimate of the mean of the states.
% P_pfekfQMC = P0*ones(T,1);             % EKF estimate of the variance of the states.
% 
% xpfqmcekf(1,:) = x(1) ;
% 
% disp(' ');
% 
%                                 % Initialize timer for benchmarking
% fprintf('QSIR-EKF');
% fprintf('\n')
% tic;
% for t=2:T,    
% 
%   
%   % PREDICTION STEP:
%   % ================ 
%   % We use the EKF as proposal.
%   for i=1:N,
%     muPred_pfekfQMC(t) = feval('ffun',xparticle_pfekfQMC(t-1,i),t);
%     Jx = 0.5;                                 % Jacobian for ffun.
%     PPred_pfekfQMC(t) = Q_pfekf + Jx*Pparticle_pfekfQMC(t-1,i)*Jx'; 
%     yPredTmp = feval('hfun',muPred_pfekfQMC(t),t);
%     if t<=30,
%       Jy = 2*0.2*muPred_pfekfQMC(t);                     % Jacobian for hfun.
%     else
%       Jy = 0.5;
%     end;
%     M = R_pfekf + Jy*PPred_pfekfQMC(t)*Jy';                  % Innovations covariance.
%     K = PPred_pfekfQMC(t)*Jy'*inv(M);                  % Kalman gain.
%     mu_pfekfQMC(t,i) = muPred_pfekfQMC(t) + K*(y(t)-yPredTmp); % Mean of proposal.
%     P_pfekfQMC(t) = PPred_pfekfQMC(t) - K*Jy*PPred_pfekfQMC(t);          % Variance of proposal.
%     xparticlePred_pfekfQMC(t,i) = mu_pfekfQMC(t,i) + sqrtm(P_pfekfQMC(t))*randn(1,1);
%     PparticlePred_pfekfQMC(t,i) = P_pfekfQMC(t);
%   end;
% 
%  
%   % EVALUATE IMPORTANCE WEIGHTS:
%   % ============================
%   % For our choice of proposal, the importance weights are give by:  
%   for i=1:N,
%     yPred_pfekfQMC(t,i) = feval('hfun',xparticlePred_pfekfQMC(t,i),t);        
%     lik = inv(sqrt(sigma)) * exp(-0.5*inv(sigma)*((y(t)-yPred_pfekfQMC(t,i))^(2)))+1e-99;
%     prior = ((xparticlePred_pfekfQMC(t,i)-xparticle_pfekfQMC(t-1,i))^(g1-1)) ...
% 		 * exp(-g2*(xparticlePred_pfekfQMC(t,i)-xparticle_pfekfQMC(t-1,i)));
%     proposal = inv(sqrt(PparticlePred_pfekfQMC(t,i))) * ...
% 	       exp(-0.5*inv(PparticlePred_pfekfQMC(t,i)) *((xparticlePred_pfekfQMC(t,i)-mu_pfekfQMC(t,i))^(2)));
%     w(t,i) = lik*prior/proposal;      
%   end;  
%   w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.
%   
%   xpfqmcekf(t,1)=xparticlePred_pfekfQMC(t,:)*(w(t,:))';      
%   
%   % SELECTION STEP:
%        
%   [xparticle_pfekfQMC(t,:),w(t,:),Pparticle_pfekfQMC(t,:)] = feval('qmcmultiresample_p',xparticlePred_pfQMC(t,:),w(t,:),Pparticle_pfekfQMC(t,:),N,d);  
%   w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.
% 
%  
%    
% end;   % End of t loop.
% 
% time_pfqmcekf(j) = toc;
% 
% 
% %%%%%%%%%%%%%%%  PERFORM SEQUENTIAL MONTE CARLO  %%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%  ======== UKF proposal ========  %%%%%%%%%%%%%%%%%%%%%
% 
% % INITIALISATION:
% % ==============
% xparticle_pfukf = ones(T,N);        % These are the particles for the estimate
%                                     % of x. Note that there's no need to store
%                                     % them for all t. We're only doing this to
%                                     % show you all the nice plots at the end.
% Pparticle_pfukf = P0*ones(T,N);     % Particles for the covariance of x.
% xparticlePred_pfukf = ones(T,N);    % One-step-ahead predicted values of the states.
% PparticlePred_pfukf = ones(T,N);    % One-step-ahead predicted values of P.
% yPred_pfukf = ones(T,N);            % One-step-ahead predicted values of y.
% w = ones(T,N);                      % Importance weights.
% mu_pfukf = ones(T,1);               % EKF estimate of the mean of the states.
% 
% error=0;
% xpfukf(1,:) = x(1) ;
% disp(' ');
% 
% tic;
%   fprintf('PF-UKF');
%   fprintf('\n')
% for t=2:T,    
% 
%   
%   % PREDICTION STEP:
%   % ================ 
%   % We use the UKF as proposal.
%   for i=1:N,
%     % Call Unscented Kalman Filter
%     [mu_pfukf(t,i),PparticlePred_pfukf(t,i)]=ukf(xparticle_pfukf(t-1,i),Pparticle_pfukf(t-1,i),[],Q_pfukf,'ukf_ffun',y(t),R_pfukf,'ukf_hfun',t,alpha,beta,kappa);
%     xparticlePred_pfukf(t,i) = mu_pfukf(t,i) + sqrtm(PparticlePred_pfukf(t,i))*randn(1,1);
%   end;
% 
%   % EVALUATE IMPORTANCE WEIGHTS:
%   % ============================
%   % For our choice of proposal, the importance weights are give by:  
%   for i=1:N,
%     yPred_pfukf(t,i) = feval('hfun',xparticlePred_pfukf(t,i),t);        
%     lik = inv(sqrt(sigma)) * exp(-0.5*inv(sigma)*((y(t)-yPred_pfukf(t,i))^(2)))+1e-99;
%     prior = ((xparticlePred_pfukf(t,i)-xparticle_pfukf(t-1,i))^(g1-1)) ...
% 		 * exp(-g2*(xparticlePred_pfukf(t,i)-xparticle_pfukf(t-1,i)));
%     proposal = inv(sqrt(PparticlePred_pfukf(t,i))) * ...
% 	       exp(-0.5*inv(PparticlePred_pfukf(t,i)) *((xparticlePred_pfukf(t,i)-mu_pfukf(t,i))^(2)));
%     w(t,i) = lik*prior/proposal;      
%   end;  
%   w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.
%   
%   % SELECTION STEP:
%   % ===============
%   % Here, we give you the choice to try three different types of
%   % resampling algorithms. Note that the code for these algorithms
%   % applies to any problem!
%   outIndex = systematicR(1:N,w(t,:)');      % Systematic resampling.
%   xparticle_pfukf(t,:) = xparticlePred_pfukf(t,outIndex); % Keep particles with
%                                               % resampled indices.
%   Pparticle_pfukf(t,:) = PparticlePred_pfukf(t,outIndex);  
%   
%   w(t,:) = 1/N ;
%   xpfukf(t,1) = xparticle_pfukf(t,:)*w(t,:)';
%   
% end;   % End of t loop.
% 
% time_pfukf(j) = toc;
% 
% 
% 
% 
% 
% % % % % % % % % %%%%%%%%%%%%%%  PERFORM QSIR  %%%%%%%%%%%%%%%%
% % % % % % % % % %%%%%%%%%%%%%%  UKF proposal  %%%%%%%%%%%%%%%%
% 
% % INITIALISATION:
% % ==============
% xparticle_pfukfQMC = ones(T,N);        % These are the particles for the estimate
%                                       % of x. Note that there's no need to store
%                                       % them for all t. We're only doing this to
%                                       % show you all the nice plots at the end.
% Pparticle_pfukfQMC = P0*ones(T,N);     % Particles for the covariance of x.
% xparticlePred_pfukfQMC = ones(T,N);    % One-step-ahead predicted values of the states.
% PparticlePred_pfukfQMC = ones(T,N);    % One-step-ahead predicted values of P.
% yPred_pfukfQMC = ones(T,N);            % One-step-ahead predicted values of y.
% w = ones(T,N);                        % Importance weights.
% mu_pfukfQMC = ones(T,1);               % EKF estimate of the mean of the states.
% 
% error=0;
% xpfqmcukf(1,:) = x(1) ;
% disp(' ');
% 
% tic;
%   fprintf('QSIR-UKF');
%   fprintf('\n')
% for t=2:T,    
% 
%   
%   % PREDICTION STEP:
%   % ================ 
%   % We use the UKF as proposal.
%   for i=1:N,
%     % Call Unscented Kalman Filter
%     [mu_pfukfQMC(t,i),PparticlePred_pfukfQMC(t,i)]=ukf(xparticle_pfukfQMC(t-1,i),Pparticle_pfukfQMC(t-1,i),[],Q_pfukf,'ukf_ffun',y(t),R_pfukf,'ukf_hfun',t,alpha,beta,kappa);
%     xparticlePred_pfukfQMC(t,i) = mu_pfukfQMC(t,i) + sqrtm(PparticlePred_pfukfQMC(t,i))*randn(1,1);
%   end;    
% 
%   % EVALUATE IMPORTANCE WEIGHTS:
%   % ============================
%   % For our choice of proposal, the importance weights are give by:  
%   for i=1:N,
%     yPred_pfukfQMC(t,i) = feval('hfun',xparticlePred_pfukfQMC(t,i),t);        
%     lik = inv(sqrt(sigma)) * exp(-0.5*inv(sigma)*((y(t)-yPred_pfukfQMC(t,i))^(2)))+1e-99;
%     prior = ((xparticlePred_pfukfQMC(t,i)-xparticle_pfukfQMC(t-1,i))^(g1-1)) ...
% 		 * exp(-g2*(xparticlePred_pfukfQMC(t,i)-xparticle_pfukfQMC(t-1,i)));
%     proposal = inv(sqrt(PparticlePred_pfukfQMC(t,i))) * ...
% 	       exp(-0.5*inv(PparticlePred_pfukfQMC(t,i)) *((xparticlePred_pfukfQMC(t,i)-mu_pfukfQMC(t,i))^(2)));
%     w(t,i) = lik*prior/proposal;      
%   end;  
%   w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.
%   
%   xpfqmcukf(t,1)=xparticlePred_pfukfQMC(t,:)*(w(t,:))';      
%   
%   % SELECTION STEP:       
%   [xparticle_pfukfQMC(t,:),w(t,:),Pparticle_pfukfQMC(t,:) ] = feval('qmcmultiresample_p',xparticlePred_pfukfQMC(t,:),w(t,:),Pparticle_pfukfQMC(t,:),N,d);  
%   w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.
% 
%   
% end;   % End of t loop.
% 
% time_pfqmcukf(j) = toc;


%%%%%%%%%%%%%%%  PERFORM SQMC   %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%                 %%%%%%%%%%%%%%%%%%%%%

% INITIALISATION:
% ==============
xparticle_SQMC = ones(T,N);        % These are the particles for the estimate
                                    % of x. Note that there's no need to store
                                    % them for all t. We're only doing this to
                                    % show you all the nice plots at the end.
Pparticle_SQMC = P0*ones(T,N);     % Particles for the covariance of x.
xparticlePred_SQMC = ones(T,N);    % One-step-ahead predicted values of the states.
PparticlePred_SQMC = ones(T,N);    % One-step-ahead predicted values of P.
yPred_SQMC = ones(T,N);            % One-step-ahead predicted values of y.
w = ones(T,N);                      % Importance weights.

error=0;

disp(' ');

tic;
fprintf('SQMC');
fprintf('\n')
  
xparticle_SQMC(1,:) =feval('qmcinterval',0,2,d,N); 
for i=1:N,
  yPred_SQMC(1,i) = feval('hfun',xparticle_SQMC(1,i),1);
 lik = inv(sqrt(sigma)) * exp(-0.5*inv(sigma)*((y(1)-yPred_SQMC(1,i))^(2))) ...
	  + 1e-99; % Deal with ill-conditioning.
  w(1,i) = lik;
end;
w(1,:)=w(1,:)/sum(w(1,:));
xpfqmc(1,1) = xparticle_SQMC(1,:)*(w(1,:))';
likPre = zeros(N,1);
for t=2:T,    
 
  % PREDICTION STEP:
  % ================ 
xparticle_SQMC(t,:)=feval('qmcinterval',0,10,d,N) ;
for i=1:N,
  xparticlePred_SQMC(t,i) = feval('ffun',xparticle_SQMC(t-1,i),t)+gengamma(g1,g2);
end;
for l=1:N,
    for i=1:N,
       likPre(l,i) = w(t-1,i).* ( ( xparticlePred_SQMC(t,i)-xparticle_SQMC(t,l) ).^(g1-1) ) ...
		 * exp( -g2 * ( xparticlePred_SQMC(t,i)-xparticle_SQMC(t,l)) ) ;
    end;
    likPre(l,1) = sum(likPre(l,:));
end;
	 
  % EVALUATE IMPORTANCE WEIGHTS:
  % ============================
  % For our choice of proposal, the importance weights are give by:  
  for i=1:N,
    yPred_SQMC(t,i) = feval('hfun',xparticle_SQMC(t,i),t);        
    lik =inv(sqrt(sigma)) * exp(-0.5*inv(sigma)*((y(t)-yPred_SQMC(t,i))^(2))) ...
	  + 1e-99; % Deal with ill-conditioning.
%     w(t,i) = lik*w(t-1,i);  w(t-1,i).*  
    w(t,i) = lik*likPre(i,1);  
  end;  
  w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.
 
  xsqmc(t,1) = xparticle_SQMC(t,:)*(w(t,:))';  
  
end;   % End of t loop.

time_sqmc(j) = toc;





%%%%%%%%%%%%%%  PERFORM SQMC  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%  ============= UKF proposal =============  %%%%%%%%%%%%%%%%

% INITIALISATION:
% ==============
xparticle_SQMC_ukf = ones(T,N);        % These are the particles for the estimate
                                      % of x. Note that there's no need to store
                                      % them for all t. We're only doing this to
                                      % show you all the nice plots at the end.
sqmc_P_ukf = P0*ones(T);     % Particles for the covariance of x.
xparticlePred_SQMC_ukf = ones(T,N);    % One-step-ahead predicted values of the states.
PparticlePred_SQMC_ukf = ones(T,N);    % One-step-ahead predicted values of P.
yPred_SQMC_ukf = ones(T,N);            % One-step-ahead predicted values of y.
w = ones(T,N);                        % Importance weights.
mu_pfukfMC = ones(T,1);               % EKF estimate of the mean of the states.
a0 = x(1) - 1;
b0 = x(1) + 1;

disp(' ');
tic;
  fprintf('SQMC-UKF');
  fprintf('\n')
  
  xparticle_SQMC_ukf(1,:) =feval('qmcinterval',a0,b0,d,N); 
for i=1:N,
  yPred_SQMC_ukf(1,i) = feval('hfun',xparticle_SQMC_ukf(1,i),1);
 lik = inv(sqrt(sigma)) * exp(-0.5*inv(sigma)*((y(1)-yPred_SQMC_ukf(1,i))^(2))) ...
	  + 1e-99; % Deal with ill-conditioning.
  w(1,i) = lik;
end;
w(1,:)=w(1,:)/sum(w(1,:));
xsqmcukf(1,1) = xparticle_SQMC_ukf(1,:)*(w(1,:))';
likPre = zeros(N,1);
sqmc_mu_ukf(1) = xsqmcukf(1,1);

for t=2:T,    
 
  % PREDICTION STEP:
  % ================  
[sqmc_mu_ukf(t),sqmc_P_ukf(t)]=ukf(sqmc_mu_ukf(t-1),sqmc_P_ukf(t-1),[],Q_pfukf,'ukf_ffun',y(t),R_pfukf,'ukf_hfun',t,alpha,beta,kappa);
sn = sqmc_P_ukf(t);
L = 1.66023*sn/sqrt(N);
a(t) = sqmc_mu_ukf(t)+ L;
b(t) = sqmc_mu_ukf(t)- L;
xparticle_SQMC_ukf(t,:)=feval('qmcinterval',a(t),b(t),d,N) ;
%  xparticle_QMC(:,:)=feval('qmc',d,N) ;
%  r = sqrt(sqmc_P_ukf(t));
% for i=1:N,
%    xparticle_SQMC_ukf(t,i) = norminv(max( min(xparticle_QMC(1,i),0.95),max(xparticle_QMC(1,i),0.05) ),0,sqmc_P_ukf(t));
%  end;
% xparticle_SQMC_ukf(t,:) = sqmc_mu_ukf(t) + r.*xparticle_SQMC_ukf(t,:);

for i=1:N,
  xparticlePred_SQMC_ukf(t,i) = feval('ffun',xparticle_SQMC_ukf(t-1,i),t)+gengamma(g1,g2);
end;
for l=1:N,
    for i=1:N,
       likPre(l,i) = w(t-1,i).* ( ( xparticlePred_SQMC_ukf(t,i)-xparticle_SQMC_ukf(t,l) ).^(g1-1) ) ...
		 * exp( -g2 * ( xparticlePred_SQMC_ukf(t,i)-xparticle_SQMC_ukf(t,l)) ) ;
    end;
    likPre(l,1) = sum(likPre(l,:));
end;

  % EVALUATE IMPORTANCE WEIGHTS:
  % ============================
  % For our choice of proposal, the importance weights are give by:  
  for i=1:N,
    yPred_SQMC_ukf(t,i) = feval('hfun',xparticle_SQMC_ukf(t,i),t);        
    lik = inv(sqrt(sigma))*exp(-0.5*inv(sigma)*((y(t)-yPred_SQMC_ukf(t,i))^(2))) ...
	  + 1e-99; % Deal with ill-conditioning.
%     w(t,i) = lik*w(t-1,i);  w(t-1,i).*  
    w(t,i) = lik*likPre(i,1);  
  end;  
  w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.  
  xsqmcukf(t,1) = xparticle_SQMC_ukf(t,:)*(w(t,:))'; 
%   P_ukf(t) = (xsqmc_ukf(t,1)-mu_ukf(t)).^2 ;
  sqmc_mu_ukf(t) = xsqmcukf(t,1);
  sqmc_P_ukf(t) = P0 ;
   
 
end;   % End of t loop.

time_sqmcukf(j) = toc;



%%%%%%%%%%%%%%%%%%%%%  PLOT THE RESULTS  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  ================  %%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-- CALCULATE PERFORMANCE --%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rmsError_ekf(j)     = sqrt(inv(T)*sum((x-mu_ekf).^(2)));
rmsError_ukf(j)     = sqrt(inv(T)*sum((x-mu_ukf).^(2)));
rmsError_pf(j)      = sqrt(inv(T)*sum((x'-xpf').^(2)));
rmsError_pfqmc(j)    = sqrt(inv(T)*sum((x'-xpfqmc').^(2)));
rmsError_pfekf(j)   = sqrt(inv(T)*sum((x'-xpfekf').^(2)));
rmsError_pfekfqmc(j) = sqrt(inv(T)*sum((x'-xpfqmcekf').^(2)));
rmsError_pfukf(j)   = sqrt(inv(T)*sum((x'-xpfukf').^(2)));
rmsError_pfukfqmc(j) = sqrt(inv(T)*sum((x'-xpfqmcukf').^(2)));
rmsError_sqmc(j)   = sqrt(inv(T)*sum((x'-xsqmc').^(2)));
rmsError_sqmcukf(j) = sqrt(inv(T)*sum((x'-xsqmcukf').^(2)));

disp(' ');
disp('Root mean square (RMS) errors');
disp('-----------------------------');
disp(' ');
disp(['EKF          = ' num2str(rmsError_ekf(j))]);
disp(['UKF          = ' num2str(rmsError_ukf(j))]);
disp(['PF           = ' num2str(rmsError_pf(j))]);
disp(['QSIR     = ' num2str(rmsError_pfqmc(j))]);
disp(['PF-EKF       = ' num2str(rmsError_pfekf(j))]);
disp(['QSIR-EKF  = ' num2str(rmsError_pfekfqmc(j))]);
disp(['PF-UKF       = ' num2str(rmsError_pfukf(j))]);
disp(['QSIR-UKF  = ' num2str(rmsError_pfukfqmc(j))]);
disp(['SQMC       = ' num2str(rmsError_sqmc(j))]);
disp(['SQMC-UKF  = ' num2str(rmsError_sqmcukf(j))]);

disp(' ');
disp(' ');
disp('Execution time  (seconds)');
disp('-------------------------');
disp(' ');
disp(['PF           = ' num2str(time_pf(j))]);
disp(['QSIR      = ' num2str(time_pfqmc(j))]);
disp(['PF-EKF       = ' num2str(time_pfekf(j))]);
disp(['QSIR-EKF  = ' num2str(time_pfqmcekf(j))]);
disp(['PF-UKF       = ' num2str(time_pfukf(j))]);
disp(['QSIR-UKF  = ' num2str(time_pfqmcukf(j))]);
disp(['SQMC       = ' num2str(time_sqmc(j))]);
disp(['SQMC-UKF  = ' num2str(time_sqmcukf(j))]);
disp(' ');

drawnow;

%*************************************************************************

end    % Main loop (for j...)

% calculate mean of RMSE errors
mean_RMSE_ekf     = mean(rmsError_ekf);
mean_RMSE_ukf     = mean(rmsError_ukf);
mean_RMSE_pf      = mean(rmsError_pf);
mean_RMSE_pfqmc    = mean(rmsError_pfqmc);
mean_RMSE_pfekf   = mean(rmsError_pfekf);
mean_RMSE_pfqmcekf = mean(rmsError_pfekfqmc);
mean_RMSE_pfukf   = mean(rmsError_pfukf);
mean_RMSE_pfqmcukf = mean(rmsError_pfukfqmc);
mean_RMSE_sqmc   = mean(rmsError_sqmc);
mean_RMSE_sqmcukf = mean(rmsError_sqmcukf);

% calculate variance of RMSE errors
var_RMSE_ekf     = var(rmsError_ekf);
var_RMSE_ukf     = var(rmsError_ukf);
var_RMSE_pf      = var(rmsError_pf);
var_RMSE_pfqmc    = var(rmsError_pfqmc);
var_RMSE_pfekf   = var(rmsError_pfekf);
var_RMSE_pfqmcekf = var(rmsError_pfekfqmc);
var_RMSE_pfukf   = var(rmsError_pfukf);
var_RMSE_pfqmcukf = var(rmsError_pfukfqmc);
var_RMSE_sqmc   = var(rmsError_sqmc);
var_RMSE_sqmcukf = var(rmsError_sqmcukf);

% calculate mean of execution time

mean_time_pf      = mean(time_pf);
mean_time_pfqmc    = mean(time_pfqmc);
mean_time_pfekf   = mean(time_pfekf);
mean_time_pfqmcekf = mean(time_pfqmcekf);
mean_time_pfukf   = mean(time_pfukf);
mean_time_pfqmcukf = mean(time_pfqmcukf);
mean_time_sqmc   = mean(time_sqmc);
mean_time_sqmcukf = mean(time_sqmcukf);


% display final results

disp(' ');
disp(' ');
disp('************* FINAL RESULTS *****************');
disp(' ');
disp('RMSE : mean and variance');
disp('---------');
disp(' ');
disp(['EKF          = ' num2str(mean_RMSE_ekf) ' (' num2str(var_RMSE_ekf) ')']);
disp(['UKF          = ' num2str(mean_RMSE_ukf) ' (' num2str(var_RMSE_ukf) ')']);
disp(['PF           = ' num2str(mean_RMSE_pf) ' (' num2str(var_RMSE_pf) ')']);
disp(['QSIR      = ' num2str(mean_RMSE_pfqmc) ' (' num2str(var_RMSE_pfqmc) ')']);
disp(['PF-EKF       = ' num2str(mean_RMSE_pfekf) ' (' num2str(var_RMSE_pfekf) ')']);
disp(['QSIR-EKF  = ' num2str(mean_RMSE_pfqmcekf) ' (' num2str(var_RMSE_pfqmcekf) ')']);
disp(['PF-UKF       = ' num2str(mean_RMSE_pfukf) ' (' num2str(var_RMSE_pfukf) ')']);
disp(['QSIR_UKF  = ' num2str(mean_RMSE_pfqmcukf) ' (' num2str(var_RMSE_pfqmcukf) ')']);
disp(['SQMC       = ' num2str(mean_RMSE_sqmc) ' (' num2str(var_RMSE_sqmc) ')']);
disp(['SQMC_UKF  = ' num2str(mean_RMSE_sqmcukf) ' (' num2str(var_RMSE_sqmcukf) ')']);

disp(' ');
disp(' ');
disp('Execution time  (seconds)');
disp('-------------------------');
disp(' ');
disp(['PF           = ' num2str(mean_time_pf)]);
disp(['QSIR      = ' num2str(mean_time_pfqmc)]);
disp(['PF-EKF       = ' num2str(mean_time_pfekf) ]);
disp(['QSIR-EKF  = ' num2str(mean_time_pfqmcekf)]);
disp(['PF-UKF       = ' num2str(mean_time_pfukf) ]);
disp(['QSIR_UKF  = ' num2str(mean_time_pfqmcukf)]);
disp(['SQMC       = ' num2str(mean_time_sqmc)]);
disp(['SQMC_UKF  = ' num2str(mean_time_sqmcukf)]);
disp(' ');

%*************************************************************************

break;

% This is an alternative way of plotting the 3D stuff:
% Somewhere in between lies the best way!
figure(3)
clf;
support=[-1:.1:2];
NN=50;
extPlot=zeros(10*61,1);
for t=6:5:T,
  [r,d]=hist(yPred_pf(t,:),support);
  r=r/sum(r);
  for i=1:1:61,
    for j=1:1:NN,
      extPlot(NN*i-NN+1:i*NN) = r(i);
    end;
  end;
  d= linspace(-5,25,length(extPlot));
  plot3(d,t*ones(size(extPlot)),extPlot,'r')
  hold on;
end;



















