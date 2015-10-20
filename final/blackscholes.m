% PURPOSE : Demonstrate the differences between the following
% filters on an options pricing problem.
%           
%           1) Extended Kalman Filter  (EKF)
%           2) Unscented Kalman Filter (UKF)
%           3) Particle Filter         (PF)
%           4) PF with EKF proposal    (PFEKF)
%           5) PF with UKF proposal    (PFUKF)


% For more details refer to:

% AUTHORS  : Nando de Freitas      (jfgf@cs.berkeley.edu)
%            Rudolph van der Merwe (rvdmerwe@ece.ogi.edu)
%            + We re-used a bit of code by Mahesan Niranjan.             
% DATE     : 17 August 2000

clear all;
echo off;
path('./ukf',path);
path('./data',path);


% INITIALISATION AND PARAMETERS:
% ==============================
doPlot = 0;                 % 1 plot online. 0 = only plot at the end.
g1 = 3;                     % Paramater of Gamma transition prior.
g2 = 2;                     % Parameter of Gamman transition prior.
                            % Thus mean = 3/2 and var = 3/4.
T = 204;                    % Number of time steps.
R = diag([1e-5 1e-5]);      % EKF's measurement noise variance. 
Q = diag([1e-7 1e-5]);      % EKF's process noise variance.
P01 = 0.1;                  % EKF's initial variance of the
                            % interest rate.
P02 = 0.1;                  % EKF's initial variance of the volatility.

N = 10;                     % Number of particles.
optionNumber = 1;           % There are 5 pairs of options.
resamplingScheme = 1;       % The possible choices are
                            % systematic sampling (2),
                            % residual (1)
                            % and multinomial (3). 
                            % They're all O(N) algorithms. 

P01_ukf = 0.1;
P02_ukf = 0.1;
			    
Q_ukf = Q;
R_ukf = R;
			    
initr = .01;
initsig = .15;

Q_pfekf = 10*1e-5*eye(2);
R_pfekf = 1e-6*eye(2);

Q_pfukf = Q_pfekf;
R_pfukf = R_pfekf;
			    
alpha = 1;                     % UKF : point scaling parameter
beta  = 2;                     % UKF : scaling parameter for higher order terms of Taylor series expansion 
kappa = 1;                     % UKF : sigma point selection scaling parameter (best to leave this = 0)

no_of_experiments = 1;         % Number of times the experiment is
                               % repeated (for statistical purposes).

% DATA STRUCTURES FOR RESULTS
% ===========================

errorcTrivial = zeros(no_of_experiments,1);
errorpTrivial = errorcTrivial;
errorcEKF     = errorcTrivial;
errorpEKF     = errorcTrivial;
errorcUKF     = errorcTrivial;
errorpUKF     = errorcTrivial;
errorcPF      = errorcTrivial;
errorpPF      = errorcTrivial;
errorcPFEKF   = errorcTrivial;
errorpPFEKF   = errorcTrivial;
errorcPFUKF   = errorcTrivial;
errorpPFUKF   = errorcTrivial;


% LOAD THE DATA:
% =============
fprintf('\n')
fprintf('Loading the data')
fprintf('\n')
load c2925.prn;         load p2925.prn;
load c3025.prn;         load p3025.prn;
load c3125.prn;         load p3125.prn;
load c3225.prn;         load p3225.prn;
load c3325.prn;         load p3325.prn;
X=[2925; 3025; 3125; 3225; 3325];
[d1,i1]=sort(c2925(:,1));  Y1=c2925(i1,:);      Z1=p2925(i1,:);
[d2,i2]=sort(c3025(:,1));  Y2=c3025(i2,:);      Z2=p3025(i2,:);
[d3,i3]=sort(c3125(:,1));  Y3=c3125(i3,:);      Z3=p3125(i3,:);
[d4,i4]=sort(c3225(:,1));  Y4=c3225(i4,:);      Z4=p3225(i4,:);
[d5,i5]=sort(c3325(:,1));  Y5=c3325(i5,:);      Z5=p3325(i5,:);
d=Y1(:,1); 
% d - date to maturity.
St(1,:) = Y1(:,3)';   C(1,:) = Y1(:,2)';  P(1,:) = Z1(:,2)';
St(2,:) = Y2(:,3)';   C(2,:) = Y2(:,2)';  P(2,:) = Z2(:,2)';
St(3,:) = Y3(:,3)';   C(3,:) = Y3(:,2)';  P(3,:) = Z3(:,2)';
St(4,:) = Y4(:,3)';   C(4,:) = Y4(:,2)';  P(4,:) = Z4(:,2)';
St(5,:) = Y5(:,3)';   C(5,:) = Y5(:,2)';  P(5,:) = Z5(:,2)';
% St - Stock price.
% C - Call option price.
% P - Put Option price.
% X - Strike price.
% Normalise with respect to the strike price:
for i=1:5
   Cox(i,:) = C(i,:) / X(i);
   Sox(i,:) = St(i,:) / X(i);
   Pox(i,:) = P(i,:) / X(i);
end
Cpred=zeros(T,5);
Ppred=zeros(T,5);

% PLOT THE LOADED DATA:
% ====================
figure(1)
clf;
plot(Cox');
ylabel('Call option prices','fontsize',15);
xlabel('Time to maturity','fontsize',15);
fprintf('\n')
fprintf('Press a key to continue')  
pause;
fprintf('\n')
fprintf('\n')
fprintf('Training has started')
fprintf('\n')

% SPECIFY THE INPUTS AND OUTPUTS
% ==============================
ii=optionNumber;   % Only one call price. Change 1 to 3, etc. for other prices.
X = X(ii,1);
St = Sox(ii,1:T);
C = Cox(ii,1:T);
P = Pox(ii,1:T);
counter=1:1:T;
tm = (224*ones(size(counter))-counter)/260;
u = [St' tm']';
y = [C' P']'; % Call and put prices.


% MAIN LOOP
% =========
for expr=1:no_of_experiments,

  rand('state',sum(100*clock));   % Shuffle the pack!
  randn('state',sum(100*clock));   % Shuffle the pack!  
  
%%%%%%%%%%%%%%%  PERFORM EKF and UKF ESTIMATION  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%  ==============================  %%%%%%%%%%%%%%%%%%%%%

% INITIALISATION:
% ==============
mu_ekf = ones(2,T);       % EKF estimate of the mean of the states.
Inn = ones(2,2,T);        % Innovations Covariance.
Inn_ukf = Inn;
mu_ekf(1,1) = initr;
mu_ekf(2,1) = initsig;
P_ekf = ones(2,2,T);      % EKF estimate of the variance of the states.
for t=1:T
  P_ekf(:,:,t)= diag([P01 P02]);
end;
mu_ukf = mu_ekf;        % UKF estimate of the mean of the states.
P_ukf = P_ekf;          % UKF estimate of the variance of the states.

yPred = ones(2,T);      % One-step-ahead predicted values of y.
yPred_ukf = yPred;
mu_ekfPred = mu_ekf;    % EKF O-s-a estimate of the mean of the states.
PPred =eye(2);          % EKF O-s-a estimate of the variance of the states.
disp(' ');

for t=2:T,    
  fprintf('EKF & UKF : t = %i / %i  \r',t,T);
  fprintf('\n')
  
  % EKF PREDICTION STEP:
  % ==================== 
  mu_ekfPred(:,t) = feval('bsffun',mu_ekf(:,t-1),t);
  Jx = eye(2);  % Jacobian for bsffun.  
  PPred = Q + Jx*P_ekf(:,:,t-1)*Jx'; 
  
  % EKF CORRECTION STEP:
  % ====================
  yPred(:,t) = feval('bshfun',mu_ekfPred(:,t),u(:,t),t);

  % COMPUTE THE JACOBIAN:
  St  = u(1,t);            % Index price.
  tm  = u(2,t);            % Time to maturity.
  r   = mu_ekfPred(1,t);   % Risk free interest rate.
  sig = mu_ekfPred(2,t);   % Volatility.  
  d1 = (log(St) + (r+0.5*(sig^2))*tm ) / (sig * (tm^0.5));
  d2 = d1 - sig * (tm^0.5);  
  % Differentials of call price
  dcsig = St * sqrt(tm) * exp(-d1^2) / sqrt(2*pi);
  dcr   = tm * exp(-r*tm) * normcdf(d2);
  % Differentials of put price
  dpsig = dcsig;
  dpr   = -tm * exp(-r*tm) * normcdf(-d2);
  Jy = [dcr dpr; dcsig dpsig]'; % Jacobian for bshfun.

  % APPLY THE EKF UPDATE EQUATIONS:
  M = R + Jy*PPred*Jy';                 % Innovations covariance.
  Inn(:,:,t)=M;
  K = PPred*Jy'*inv(M);                 % Kalman gain.
  mu_ekf(:,t) = mu_ekfPred(:,t) + K*(y(:,t)-yPred(:,t));
  P_ekf(:,:,t) = PPred - K*Jy*PPred;
  
  % Full Unscented Kalman Filter step
  % =================================
  [mu_ukf(:,t),P_ukf(:,:,t),zab1,zab2,yPred_ukf(:,t),inov_ukf,Inn_ukf(:,:,t),K_ukf]=ukf(mu_ukf(:,t-1),P_ukf(:,:,t-1),u(:,t),Q_ukf,'ukf_bsffun',y(:,t),R_ukf,'ukf_bshfun',t,alpha,beta,kappa);
  
end;   % End of t loop.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-- CALCULATE PERFORMANCE --%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(3)
clf;
subplot(211)
p1=plot(1:T,mu_ekf(1,:),'r','linewidth',2);hold on;
p2=plot(1:T,mu_ukf(1,:),'b','linewidth',2);hold off;
legend([p1 p2],'ekf','ukf');
ylabel('Interest rate','fontsize',15)
subplot(212)
p1=plot(1:T,mu_ekf(2,:),'r','linewidth',2);hold on;
p2=plot(1:T,mu_ukf(2,:),'b','linewidth',2);hold off;
ylabel('Volatility','fontsize',15);
xlabel('Time (days)','fontsize',15)
zoom on;

% Transform innovations covariance for plotting.
Inn11=zeros(1,T);
Inn22=zeros(1,T);
Pekf11=zeros(1,T);
Pekf22=zeros(1,T);
for t=1:T,
  Inn11(t)=Inn(1,1,t);
  Inn22(t)=Inn(2,2,t);
  Inn11_ukf(t)=Inn_ukf(1,1,t);
  Inn22_ukf(t)=Inn_ukf(2,2,t);
  Pekf11(t)=P_ekf(1,1,t);
  Pekf22(t)=P_ekf(2,2,t);
  Pukf11(t)=P_ukf(1,1,t);
  Pukf22(t)=P_ukf(2,2,t);
end;

figure(1)
clf;
subplot(211)
plot(1:T,y(1,:),'r--',1:T,yPred(1,:),'b','linewidth',2);
hold on;
plot(1:T,yPred(1,:)+2*sqrt(Inn11),'k',1:T,yPred(1,:)-2*sqrt(Inn11),'k')
ylabel('Call price','fontsize',15)
legend('Actual price','Prediction');
axis([0 204 0.03 .22])
title('EKF');
subplot(212)
plot(1:T,y(2,:),'r--',1:T,yPred(2,:),'b','linewidth',2);
hold on;
plot(1:T,yPred(2,:)+2*sqrt(Inn22),'k',1:T,yPred(2,:)-2*sqrt(Inn22),'k')
ylabel('Put price','fontsize',15)
xlabel('Time (days)','fontsize',15)
axis([0 204 0 .06])
zoom on;
legend('Actual price','Prediction');

figure(2)
clf;
subplot(211)
plot(1:T,y(1,:),'r--',1:T,yPred_ukf(1,:),'b','linewidth',2);
hold on;
plot(1:T,yPred_ukf(1,:)+2*sqrt(Inn11_ukf),'k',1:T,yPred_ukf(1,:)-2*sqrt(Inn11_ukf),'k')
ylabel('Call price','fontsize',15)
legend('Actual price','Prediction');
axis([0 204 0.03 .22])
title('UKF');
subplot(212)
plot(1:T,y(2,:),'r--',1:T,yPred_ukf(2,:),'b','linewidth',2);
hold on;
plot(1:T,yPred_ukf(2,:)+2*sqrt(Inn22_ukf),'k',1:T,yPred_ukf(2,:)-2*sqrt(Inn22_ukf),'k')
ylabel('Put price','fontsize',15)
xlabel('Time (days)','fontsize',15)
axis([0 204 0 .06])
zoom on;
legend('Actual price','Prediction');


%%%%%%%%%%%%%%%  PERFORM SEQUENTIAL MONTE CARLO  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%  ==============================  %%%%%%%%%%%%%%%%%%%%%

% INITIALISATION:
% ==============
xparticle_pf = ones(2,T,N);      % These are the particles for the estimate
                                 % of x. Note that there's no need to store
                                 % them for all t. We're only doing this to
                                 % show you all the nice plots at the end.
xparticlePred_pf = ones(2,T,N);    % One-step-ahead predicted values of the states.
yPred_pf = ones(2,T,N);            % One-step-ahead predicted values of y.
w = ones(T,N);                   % Importance weights.
% Initialisation:
for i=1:N,
  xparticle_pf(1,1,i) = initr; % sqrt(initr)*randn(1,1);
  xparticle_pf(2,1,i) = initsig; %sqrt(initsig)*randn(1,1);
end;
disp(' ');
 
tic;                             % Initialize timer for benchmarking

for t=2:T,    
  fprintf('PF :  t = %i / %i  \r',t,T);
  fprintf('\n')
  
  % PREDICTION STEP:
  % ================ 
  % We use the transition prior as proposal.
  for i=1:N,
    xparticlePred_pf(:,t,i) = feval('bsffun',xparticle_pf(:,t-1,i),t) + sqrtm(Q)*randn(2,1);    
  end;

  % EVALUATE IMPORTANCE WEIGHTS:
  % ============================
  % For our choice of proposal, the importance weights are give by:  
  for i=1:N,
    yPred_pf(:,t,i) = feval('bshfun',xparticlePred_pf(:,t,i),u(:,t),t);        
    lik = exp(-0.5*(y(:,t)-yPred_pf(:,t,i))'*inv(R)*(y(:,t)-yPred_pf(:,t,i)) ) + 1e-99; % Deal with ill-conditioning.
    w(t,i) = lik;    
  end;  
  w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.
  
  % SELECTION STEP:
  % ===============
  % Here, we give you the choice to try three different types of
  % resampling algorithms. Note that the code for these algorithms
  % applies to any problem!
  if resamplingScheme == 1
    outIndex = residualR(1:N,w(t,:)');        % Residual resampling.
  elseif resamplingScheme == 2
    outIndex = systematicR(1:N,w(t,:)');      % Systematic resampling.
  else  
    outIndex = multinomialR(1:N,w(t,:)');     % Multinomial resampling.  
  end;
  xparticle_pf(:,t,:) = xparticlePred_pf(:,t,outIndex); % Keep particles with
                                                    % resampled indices.
end;   % End of t loop.

time_pf = toc;    % How long did this take?

% Compute posterior mean predictions:
yPFmeanC=zeros(1,T);
yPFmeanP=zeros(1,T);
for t=1:T,
  yPFmeanC(t) = mean(yPred_pf(1,t,:));
  yPFmeanP(t) = mean(yPred_pf(2,t,:));  
end;  





figure(4)
clf;
domain = zeros(T,1);
range = zeros(T,1);
thex=[0:1e-3:20e-3];
hold on
ylabel('Time (t)','fontsize',15)
xlabel('r_t','fontsize',15)
zlabel('p(r_t|S_t,t_m,C_t,P_t)','fontsize',15)
for t=11:20:200,
  [range,domain]=hist(xparticle_pf(1,t,:),thex);
  waterfall(domain,t,range/sum(range));
end;
view(-30,80);
rotate3d on;
a=get(gca);
set(gca,'ygrid','off');

figure(5)
clf;
domain = zeros(T,1);
range = zeros(T,1);
thex=[0.1:1e-2:0.25];
hold on
ylabel('Time (t)','fontsize',15)
xlabel('r_t','fontsize',15)
zlabel('p(\sigma_t|S_t,t_m,C_t,P_t)','fontsize',15)
%v=[0 1];
%caxis(v);
for t=11:20:200,
  [range,domain]=hist(xparticle_pf(2,t,:),thex);
  waterfall(domain,t,range/sum(range));
end;
view(-30,80);
rotate3d on;
a=get(gca);
set(gca,'ygrid','off');




%%%%%%%%%%%%%%%  PERFORM SEQUENTIAL MONTE CARLO  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%  ======== EKF proposal ========  %%%%%%%%%%%%%%%%%%%%%

% INITIALISATION:
% ==============
xparticle_pfekf = ones(2,T,N);      % These are the particles for the estimate
                                    % of x. Note that there's no need to store
                                    % them for all t. We're only doing this to
                                    % show you all the nice plots at the end.
Pparticle_pfekf = cell(N,1);        % Particles for the covariance of x.
% Initialisation:
for i=1:N,
  xparticle_pfekf(1,1,i) = initr;   % sqrt(initr)*randn(1,1);
  xparticle_pfekf(2,1,i) = initsig; %sqrt(initsig)*randn(1,1);
  Pparticle_pfekf{i} = ones(2,2,T);
  for t=1:T,
    Pparticle_pfekf{i}(:,:,t)= diag([P01 P02]); 
  end;
end;  

xparticlePred_pfekf = ones(2,T,N);    % One-step-ahead predicted values of the states.
PparticlePred_pfekf = Pparticle_pfekf;    % One-step-ahead predicted values of P.

yPred_pfekf = ones(2,T,N);          % One-step-ahead predicted values of y.
w = ones(T,N);                      % Importance weights.
muPred_pfekf = ones(2,T);           % EKF O-s-a estimate of the mean of the states.
PPred_pfekf = ones(2,2);            % EKF O-s-a estimate of the variance of the states.
mu_pfekf = ones(2,T,N);             % EKF estimate of the mean of the states.
P_pfekf = ones(2,2,T);              % EKF estimate of the variance of the states.

disp(' ');

tic;                                % Initialize timer for benchmarking

for t=2:T,    
  fprintf('PF-EKF : t = %i / %i  \r',t,T);
  fprintf('\n')
  
  % PREDICTION STEP:
  % ================ 
  % We use the EKF as proposal.
  for i=1:N,
    muPred_pfekf(:,t) = feval('bsffun',xparticle_pfekf(:,t-1,i),t);
    Jx = eye(2);                                 % Jacobian for ffun.
    PPred_pfekf = Q_pfekf + Jx*Pparticle_pfekf{i}(:,:,t-1)*Jx'; 
    yPredTmp = feval('bshfun',muPred_pfekf(:,t),u(:,t),t);
    % COMPUTE THE JACOBIAN:
    St  = u(1,t);              % Index price.
    tm  = u(2,t);              % Time to maturity.
    r   = muPred_pfekf(1,t);   % Risk free interest rate.
    sig = muPred_pfekf(2,t);   % Volatility.  
    d1 = (log(St) + (r+0.5*(sig^2))*tm ) / (sig * (tm^0.5));
    d2 = d1 - sig * (tm^0.5);  
    % Differentials of call price
    dcsig = St * sqrt(tm) * exp(-d1^2) / sqrt(2*pi);
    dcr   = tm * exp(-r*tm) * normcdf(d2);
    % Differentials of put price
    dpsig = dcsig;
    dpr   = -tm * exp(-r*tm) * normcdf(-d2);
    Jy = [dcr dpr; dcsig dpsig]'; % Jacobian for bshfun.

    % APPLY THE EKF UPDATE EQUATIONS:
    M = R_pfekf + Jy*PPred_pfekf*Jy';                  % Innovations covariance.
    K = PPred_pfekf*Jy'*inv(M);                        % Kalman gain.
    mu_pfekf(:,t,i) = muPred_pfekf(:,t) + K*(y(:,t)-yPredTmp); % Mean of proposal.
    P_pfekf(:,:,t) = PPred_pfekf - K*Jy*PPred_pfekf;           % Variance of proposal.
    xparticlePred_pfekf(:,t,i) = mu_pfekf(:,t,i) + sqrtm(P_pfekf(:,:,t))*randn(2,1);
    PparticlePred_pfekf{i}(:,:,t) = P_pfekf(:,:,t);
  end;

  % EVALUATE IMPORTANCE WEIGHTS:
  % ============================
  % For our choice of proposal, the importance weights are give by:  
  for i=1:N,
    yPred_pfekf(:,t,i) = feval('bshfun',xparticlePred_pfekf(:,t,i),u(:,t),t);        
    lik = exp(-0.5*(y(:,t)-yPred_pfekf(:,t,i))'*inv(R)*(y(:,t)-yPred_pfekf(:,t,i)) ) + 1e-99;
    prior = exp(-0.5*(xparticlePred_pfekf(:,t,i)- xparticle_pfekf(:,t-1,i))'*inv(Q) * (xparticlePred_pfekf(:,t,i)-xparticle_pfekf(:,t-1,i) ))+ 1e-99;
    proposal = inv(sqrt(det(PparticlePred_pfekf{i}(:,:,t)))) * exp(-0.5*(xparticlePred_pfekf(:,t,i)-mu_pfekf(:,t,i))'*inv(PparticlePred_pfekf{i}(:,:,t)) * (xparticlePred_pfekf(:,t,i)-mu_pfekf(:,t,i)))+ 1e-99;
    w(t,i) = lik*prior/proposal;      
  end;  
  w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.
  
  % SELECTION STEP:
  % ===============
  % Here, we give you the choice to try three different types of
  % resampling algorithms. Note that the code for these algorithms
  % applies to any problem!
  if resamplingScheme == 1
    outIndex = residualR(1:N,w(t,:)');        % Residual resampling.
  elseif resamplingScheme == 2
    outIndex = systematicR(1:N,w(t,:)');      % Systematic resampling.
  else  
    outIndex = multinomialR(1:N,w(t,:)');     % Multinomial resampling.  
  end;
  xparticle_pfekf(:,t,:) = xparticlePred_pfekf(:,t,outIndex); % Keep particles with
                                                              % resampled indices.
  for i=1:N,
    Pparticle_pfekf{i} = PparticlePred_pfekf{outIndex(i)};  
  end;
end;   % End of t loop.

time_pfekf = toc;


%%%%%%%%%%%%%%%  PERFORM SEQUENTIAL MONTE CARLO  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%  ======== UKF proposal ========  %%%%%%%%%%%%%%%%%%%%%

% INITIALISATION:
% ==============
xparticle_pfukf = ones(2,T,N);      % These are the particles for the estimate
                                    % of x. Note that there's no need to store
                                    % them for all t. We're only doing this to
                                    % show you all the nice plots at the end.
Pparticle_pfukf = cell(N,1);        % Particles for the covariance of x.

%Initialization
for i=1:N,
  xparticle_pfukf(1,1,i) = initr;   % sqrt(initr)*randn(1,1);
  xparticle_pfukf(2,1,i) = initsig; % sqrt(initsig)*randn(1,1);
  Pparticle_pfukf{i} = ones(2,2,T);
  for t=1:T,
    Pparticle_pfukf{i}(:,:,t) = diag([P01_ukf P02_ukf]);
  end
end  
xparticlePred_pfukf = ones(2,T,N);       % One-step-ahead predicted values of the states.
PparticlePred_pfukf = Pparticle_pfukf;   % One-step-ahead predicted values of P.
yPred_pfukf = ones(2,T,N);               % One-step-ahead predicted values of y.
w = ones(T,N);                           % Importance weights.
muPred_pfukf = ones(2,T);                % EKF O-s-a estimate of the mean of the states.
PPred_pfukf = ones(2,2);                 % EKF O-s-a estimate of the variance of the states.
mu_pfukf = ones(2,T,N);                  % EKF estimate of the mean of the states.
P_pfukf = ones(2,2,T);                   % EKF estimate of the variance of the states.

error=0;

disp(' ');

tic;

if (1),

for t=2:T,    
  fprintf('PF-UKF : t = %i / %i  \r',t,T);
  fprintf('\n')
  
  % PREDICTION STEP:
  % ================ 
  % We use the UKF as proposal.
  for i=1:N,
    % Call Unscented Kalman Filter
    [mu_pfukf(:,t,i),P_pfukf(:,:,t)]=ukf(xparticle_pfukf(:,t-1,i),Pparticle_pfukf{i}(:,:,t-1),u(:,t),Q_pfukf,'ukf_bsffun',y(:,t),R_pfukf,'ukf_bshfun',t,alpha,beta,kappa);
    xparticlePred_pfukf(:,t,i) = mu_pfukf(:,t,i) + sqrtm(P_pfukf(:,:,t))*randn(2,1);
    PparticlePred_pfukf{i}(:,:,t) = P_pfukf(:,:,t);
    
  end;

  % EVALUATE IMPORTANCE WEIGHTS:
  % ============================
  % For our choice of proposal, the importance weights are give by:  
  for i=1:N,
    yPred_pfukf(:,t,i) = feval('bshfun',xparticlePred_pfukf(:,t,i),u(:,t),t);

    lik = exp(-0.5*(y(:,t)-yPred_pfukf(:,t,i))'*inv(R)*(y(:,t)-yPred_pfukf(:,t,i)) ) + 1e-99;

    prior = exp(-0.5*(xparticlePred_pfukf(:,t,i)- xparticle_pfukf(:,t-1,i))'*inv(Q) * (xparticlePred_pfukf(:,t,i)-xparticle_pfukf(:,t-1,i) ))+ 1e-99;    
    proposal = inv(sqrt(det(PparticlePred_pfukf{i}(:,:,t)))) * exp(-0.5*(xparticlePred_pfukf(:,t,i)-mu_pfukf(:,t,i))'*inv(PparticlePred_pfukf{i}(:,:,t)) * (xparticlePred_pfukf(:,t,i)-mu_pfukf(:,t,i)))+ 1e-99;    
    
    w(t,i) = lik*prior/proposal;      
  end;  
  w(t,:) = w(t,:)./sum(w(t,:));                % Normalise the weights.
  
  % SELECTION STEP:
  % ===============
  % Here, we give you the choice to try three different types of
  % resampling algorithms. Note that the code for these algorithms
  % applies to any problem!
  if resamplingScheme == 1
    outIndex = residualR(1:N,w(t,:)');        % Residual resampling.
  elseif resamplingScheme == 2
    outIndex = systematicR(1:N,w(t,:)');      % Systematic resampling.
  else  
    outIndex = multinomialR(1:N,w(t,:)');     % Multinomial resampling.  
  end;
  xparticle_pfukf(:,t,:) = xparticlePred_pfukf(:,t,outIndex); % Keep particles with
                                              % resampled indices.
  for i=1:N,					      
    Pparticle_pfukf{i} = PparticlePred_pfukf{outIndex(i)};
  end
  
end;   % End of t loop.

end

time_pfukf = toc;


% Compute posterior mean predictions:
yPFEKFmeanC=zeros(1,T);
yPFEKFmeanP=zeros(1,T);
for t=1:T,
  yPFEKFmeanC(t) = mean(yPred_pfekf(1,t,:));
  yPFEKFmeanP(t) = mean(yPred_pfekf(2,t,:));  
end;  
yPFUKFmeanC=zeros(1,T);
yPFUKFmeanP=zeros(1,T);
for t=1:T,
  yPFUKFmeanC(t) = mean(yPred_pfukf(1,t,:));
  yPFUKFmeanP(t) = mean(yPred_pfukf(2,t,:));  
end;  

errorcTrivial(expr) = norm(C(104:204)-C(103:203));
errorpTrivial(expr) = norm(P(104:204)-P(103:203));
errorcEKF(expr) =norm(C(104:204)-yPred(1,104:204));
errorpEKF(expr) =norm(P(104:204)-yPred(2,104:204));
errorcUKF(expr) =norm(C(104:204)-yPred_ukf(1,104:204));
errorpUKF(expr) =norm(P(104:204)-yPred_ukf(2,104:204));
errorcPF(expr) =norm(C(104:204)-yPFmeanC(104:204));
errorpPF(expr) =norm(P(104:204)-yPFmeanP(104:204));
errorcPFEKF(expr) =norm(C(104:204)-yPFEKFmeanC(104:204));
errorpPFEKF(expr) =norm(P(104:204)-yPFEKFmeanP(104:204));
errorcPFUKF(expr) =norm(C(104:204)-yPFUKFmeanC(104:204));
errorpPFUKF(expr) =norm(P(104:204)-yPFUKFmeanP(104:204));

disp(' ');
disp(['Experiment ' num2str(expr) ' of ' num2str(no_of_experiments) ' : Mean square errors sqrt(sum((errors).^2))']);
disp('------------------------------------------------------------');
disp(' ');
disp(['Trivial call   = ' num2str(errorcTrivial(expr))]);
disp(['EKF call       = ' num2str(errorcEKF(expr))]);
disp(['UKF call       = ' num2str(errorcUKF(expr))]);
disp(['PF call        = ' num2str(errorcPF(expr))]);
disp(['PF-EKF call    = ' num2str(errorcPFEKF(expr))]);
disp(['PF-UKF call    = ' num2str(errorcPFUKF(expr))]);
disp(['Trivial put    = ' num2str(errorpTrivial(expr))]);
disp(['EKF put        = ' num2str(errorpEKF(expr))]);
disp(['UKF put        = ' num2str(errorpUKF(expr))]);
disp(['PF put         = ' num2str(errorpPF(expr))]);
disp(['PF-EKF put     = ' num2str(errorpPFEKF(expr))]);
disp(['PF-UKF put     = ' num2str(errorpPFUKF(expr))]);


figure(9)
bti=20;
lw=2;
clf;
subplot(211)
p0=plot(bti:T,y(1,bti:T),'k-o','linewidth',lw); hold on;
p1=plot(bti:T,yPFmeanC(bti:T),'m','linewidth',lw);
p2=plot(bti:T,yPFEKFmeanC(bti:T),'r','linewidth',lw);
p3=plot(bti:T,yPFUKFmeanC(bti:T),'b','linewidth',lw); hold off;
ylabel('Call price','fontsize',15);
legend([p0 p1 p2 p3],'Actual price','PF prediction','PF-EKF prediction','PF-UKF prediction');
v=axis;
axis([bti T v(3) v(4)]);
subplot(212)
p0=plot(bti:T,y(2,bti:T),'k-o','linewidth',lw); hold on;
p1=plot(bti:T,yPFmeanP(bti:T),'m','linewidth',lw);
p2=plot(bti:T,yPFEKFmeanP(bti:T),'r','linewidth',lw);
p3=plot(bti:T,yPFUKFmeanP(bti:T),'b','linewidth',lw); hold off;
ylabel('Put price','fontsize',15);
legend([p0 p1 p2 p3],'Actual price','PF prediction','PF-EKF prediction','PF-UKF prediction');
xlabel('Time (days)','fontsize',15)
v=axis;
axis([bti T v(3) v(4)]);
zoom on;

end   % END OF MAIN LOOP

% CALCULATE MEAN AND VARIANCE OF EXPERIMENT RESULTS

% means
errorcTrivial_mean = mean(errorcTrivial);
errorcEKF_mean     = mean(errorcEKF);
errorcUKF_mean     = mean(errorcUKF);
errorcPF_mean      = mean(errorcPF);
errorcPFEKF_mean   = mean(errorcPFEKF);
errorcPFUKF_mean   = mean(errorcPFUKF);
errorpTrivial_mean = mean(errorpTrivial);
errorpEKF_mean     = mean(errorpEKF);
errorpUKF_mean     = mean(errorpUKF);
errorpPF_mean      = mean(errorpPF);
errorpPFEKF_mean   = mean(errorpPFEKF);
errorpPFUKF_mean   = mean(errorpPFUKF);

% variances
errorcTrivial_var = var(errorcTrivial);
errorcEKF_var     = var(errorcEKF);
errorcUKF_var     = var(errorcUKF);
errorcPF_var      = var(errorcPF);
errorcPFEKF_var   = var(errorcPFEKF);
errorcPFUKF_var   = var(errorcPFUKF);
errorpTrivial_var = var(errorpTrivial);
errorpEKF_var     = var(errorpEKF);
errorpUKF_var     = var(errorpUKF);
errorpPF_var      = var(errorpPF);
errorpPFEKF_var   = var(errorpPFEKF);
errorpPFUKF_var   = var(errorpPFUKF);

disp(' ');
disp('Mean and Variance of MSE ');
disp('-------------------------');
disp(' ');
disp(['Trivial call   : ' num2str(errorcTrivial_mean) ' (' num2str(errorcTrivial_var) ')']);
disp(['EKF call       : ' num2str(errorcEKF_mean) ' (' num2str(errorcEKF_var) ')']);
disp(['UKF call       : ' num2str(errorcUKF_mean) ' (' num2str(errorcUKF_var) ')']);
disp(['PF call        : ' num2str(errorcPF_mean) ' (' num2str(errorcPF_var) ')']);
disp(['PF-EKF call    : ' num2str(errorcPFEKF_mean) ' (' num2str(errorcPFEKF_var) ')']);
disp(['PF-UKF call    : ' num2str(errorcPFUKF_mean) ' (' num2str(errorcPFUKF_var) ')']);
disp(['Trivial put    : ' num2str(errorpTrivial_mean) ' (' num2str(errorpTrivial_var) ')']);
disp(['EKF put        : ' num2str(errorpEKF_mean) ' (' num2str(errorpEKF_var) ')']);
disp(['UKF put        : ' num2str(errorpUKF_mean) ' (' num2str(errorpUKF_var) ')']);
disp(['PF put         : ' num2str(errorpPF_mean) ' (' num2str(errorpPF_var) ')']);
disp(['PF-EKF put     : ' num2str(errorpPFEKF_mean) ' (' num2str(errorpPFEKF_var) ')']);
disp(['PF-UKF put     : ' num2str(errorpPFUKF_mean) ' (' num2str(errorpPFUKF_var) ')']);

figure(10);
subplot(211);
p1=semilogy(errorcTrivial,'k','linewidth',lw); hold on;
p2=semilogy(errorcEKF,'y','linewidth',lw);
p3=semilogy(errorcUKF,'g','linewidth',lw);
p4=semilogy(errorcPF,'m','linewidth',lw);
p5=semilogy(errorcPFEKF,'r','linewidth',lw);
p6=semilogy(errorcPFUKF,'b','linewidth',lw); hold off;
legend([p1 p2 p3 p4 p5 p6],'trivial','EKF','UKF','PF','PF-EKF','PF-UKF');
ylabel('MSE','fontsize',12);
xlabel('experiment','fontsize',12);
title('CALL Options Mean Prediction Error','fontsize',14);
subplot(212);
p1=semilogy(errorpTrivial,'k','linewidth',lw); hold on;
p2=semilogy(errorpEKF,'y','linewidth',lw);
p3=semilogy(errorpUKF,'g','linewidth',lw);
p4=semilogy(errorpPF,'m','linewidth',lw);
p5=semilogy(errorpPFEKF,'r','linewidth',lw);
p6=semilogy(errorpPFUKF,'b','linewidth',lw); hold off;
legend([p1 p2 p3 p4 p5 p6],'trivial','EKF','UKF','PF','PF-EKF','PF-UKF');
ylabel('MSE','fontsize',12);
xlabel('experiment','fontsize',12);
title('PUT Options Mean Prediction Error','fontsize',14);












