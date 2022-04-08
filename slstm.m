function out = slstm(data,train,seq,times)

%simple LSTM algorithm
%can be compared to srnnwb.m in /trend (simple RNN)
%train = training length
%seq = seq length
%times=adam algorithm iteration time

d=size(data);

tic;

if min(d)~=1||train >max(d)||seq >=train
    disp('error : data needs to be a vector or incorrect testset/sequence length');
else
    maxdata=max(data);
    data = data(1:train)/maxdata; %cutting as training size and scaling data
    
    h0=0; c0=0; %initial values
    wf1=1;wf2=1;bf=0; %initial weight and bias
    wi1=1;wi2=1;bi=0; 
    wc1=1;wc2=1;bc=0;
    wo1=1;wo2=1;bo=0;
    wy=1;by=0;
    
    h=zeros(size(data)); %hidden 
    c=zeros(size(data)); %cell
    y=zeros(size(data)); %prediction
    f=zeros(size(data)); %forget gate
    i=zeros(size(data)); %input gate
    ci = zeros(size(data)); % cell for input gate layer
    o=zeros(size(data)); %output gate
    
    f(1)=1/(1+exp(-(h0*wf1 + data(1)*wf2 + bf))); %forget gate layer
    i(1)=1/(1+exp(-(h0*wi1 + data(1)*wi2 + bi))); %input gate layer
    ci(1)=tanh(h0*wc1 + data(1)*wc2 + bc);
    c(1)=f(1)*c0 + i(1)*ci(1); %cell
    o(1)=1/(1+exp(-(h0*wo1 + data(1)*wo2 + bo))); %output gate layer
    h(1)=o(1)*c(1); %hidden layer
    y(1)=h(1)*wy + by; %prediction
    
    for j=2:length(data)
        f(j)=1/(1+exp(-(h(j-1)*wf1 + data(j)*wf2 + bf)));%forget gate layer
        i(j)=1/(1+exp(-(h(j-1)*wi1 + data(j)*wi2 + bi)));%input gate layer
        ci(j)=tanh(h(j-1)*wc1 + data(j)*wc2 + bc);
        c(j)=f(j)*c(j-1) + i(j)*ci(j);%cell
        o(j)=1/(1+exp(-(h(j-1)*wo1 + data(j)*wo2 + bo))); %output gate layer
        h(j)=o(j)*c(j);%hidden layer
        y(j)=h(j)*wy + by;%prediction
    end
    
    %Adam algorithm
    alpha=0.001; %stepsize
    beta1=0.9; %exponential decay rates for the moment estimates
    beta2=0.999; 
    mm=zeros(14,1); %initial 1st moment vector
    v=zeros(14,1); %initial 2nd moment vector
    t=0; %initialize timestep
    while t<times %try given times
          t=t+1;
          
          dcdf=zeros(3,length(data)); %par.der. dc/df where f=[w_f1,w_f2,b_f]
          dhdf=zeros(3,length(data)); %par.der. dh/df where f=[w_f1,w_f2,b_f]
          dcdf(:,1)=[0;0;0]; %first dc/df
          dhdf(:,1)=o(1)*dcdf(:,1); % first dh/df
          for k=2:length(data)
              dcdf(:,k)=f(k)*(1-f(k))*c(k-1)*(dhdf(:,k-1)*wf1+[h(k-1);data(k);1])+f(k)*dcdf(:,k-1)+i(k)*(1-i(k))*wi1*ci(k)*dhdf(:,k-1)+i(k)*(1-ci(k)^2)*wc1*dhdf(:,k-1);
              dhdf(:,k)=o(k)*(1-o(k))*wo1*c(k).*dhdf(:,k-1)+o(k).*dcdf(:,k);
          end
          dcdi=zeros(3,length(data)); %par.der. dc/di where i=[w_i1,w_i2,b_i]
          dhdi=zeros(3,length(data)); %par.der. dh/di where i=[w_i1,w_i2,b_i]
          dcdi(:,1)=i(1)*(1-i(1))*ci(1)*[0;data(1);1]; %first dc/di
          dhdi(:,1)=o(1)*dcdi(:,1); % first dh/di
          for k=2:length(data)
              dcdi(:,k)=f(k)*(1-f(k))*wf1*c(k-1)*dhdi(:,k-1)+f(k)*dcdi(:,k-1)+i(k)*(1-i(k))*ci(k)*(wi1*dhdi(:,k-1)+[h(k-1);data(k);1])+i(k)*(1-ci(k)^2)*wc1*dhdi(:,k-1);
              dhdi(:,k)=o(k)*(1-o(k))*wo1*c(k)*dhdi(:,k-1)+o(k)*dcdi(:,k);
          end
          dcdc=zeros(3,length(data)); %par.der. dc/dc where below c=[w_c1,w_c2,b_c]
          dhdc=zeros(3,length(data)); %par.der. dh/dc where below c=[w_c1,w_c2,b_c]
          dcdc(:,1)=i(1)*(1-ci(1)^2)*[0;data(1);1];
          dhdc(:,1)=o(1)*dcdc(:,1);
          for k=2:length(data)
              dcdc(:,k)=f(k)*(1-f(k))*wf1*c(k-1)*dhdc(:,k-1)+f(k)*dcdc(:,k-1)+i(k)*(1-i(k))*wi1*ci(k)*dhdc(:,k-1)+i(k)*(1-ci(k)^2)*(wc1*dhdc(:,k-1)+[h(k-1);data(k);1]);
              dhdc(:,k)=o(k)*(1-o(k))*wo1*c(k)*dhdc(:,k-1)+o(k)*dcdc(:,k);
          end
          dcdo=zeros(3,length(data)); %par.der. dc/do where o=[w_o1,w_o2,b_o]
          dhdo=zeros(3,length(data)); %par.der. dh/do where o=[w_o1,w_o2,b_o]
          dcdo(:,1)=[0;0;0];
          dhdo(:,1)=o(1)*(1-o(1))*c(1)*[0;data(1);1];
          for k=2:length(data)
              dcdo(:,k)=f(k)*(1-f(k))*wf1*c(k-1)*dhdo(:,k-1)+f(k)*dcdo(:,k-1)+i(k)*(1-i(k))*wi1*ci(k)*dhdo(:,k-1)+i(k)*(1-ci(k)^2)*wc1*dhdo(:,k-1);
              dhdo(:,k)=o(k)*(1-o(k))*c(k)*(wo1*dhdo(:,k-1)+[h(k-1);data(k);1])+o(k)*dcdo(:,k);
          end
          
          gf=[0;0;0];gi=[0;0;0];gc=[0;0;0];go=[0;0;0];gy=[0;0];
          m=train-seq;
          for j=1:m
              gf=gf+2/m*(y(seq+j-1)-data(seq+j))*wy*dhdf(:,seq+j-1); %par.der. dcost/df where f=[wf1,wf2,bf]
              gi=gi+2/m*(y(seq+j-1)-data(seq+j))*wy*dhdi(:,seq+j-1); %par.der. dcost/di where i=[wi1,wi2,bi]
              gc=gc+2/m*(y(seq+j-1)-data(seq+j))*wy*dhdc(:,seq+j-1); %par.der. dcost/dc where c=[wc1,wc2,bc]
              go=go+2/m*(y(seq+j-1)-data(seq+j))*wy*dhdo(:,seq+j-1); %par.der. dcost/do where o=[wo1,wo2,bo]
              gy=gy+2/m*(y(seq+j-1)-data(seq+j))*[h(seq+j-1);1]; %par.der. dcost/dy where y=[wy,by]
          end
          
          g=[gf;gi;gc;go;gy]; %combine all g's for easy calculation
          mm=beta1*mm+(1-beta1)*g;
          v=beta2*v+(1-beta2)*g.^2;
          mm_new=mm/(1-beta1^t);
          v_new=v/(1-beta2^t);
              
          update=alpha*mm_new./(sqrt(v_new)+ones(14,1)*10^-8); %amount of updating
          wf1=wf1-update(1);%new wf1
          wf2=wf2-update(2);%new wf2
          bf=bf-update(3); %and so on
          wi1=wi1-update(4);
          wi2=wi2-update(5);
          bi=bi-update(6);
          wc1=wc1-update(7);
          wc2=wc2-update(8);
          bc=bc-update(9);
          wo1=wo1-update(10);
          wo2=wo2-update(11);
          bo=bo-update(12);
          wy=wy-update(13);
          by=by-update(14);
      
          %check cost with new weights and biases
          f(1)=1/(1+exp(-(h0*wf1 + data(1)*wf2 + bf))); %forget gate layer
          i(1)=1/(1+exp(-(h0*wi1 + data(1)*wi2 + bi))); %input gate layer
          ci(1)=tanh(h0*wc1 + data(1)*wc2 + bc);
          c(1)=f(1)*c0 + i(1)*ci(1); %cell
          o(1)=1/(1+exp(-(h0*wo1 + data(1)*wo2 + bo))); %output gate layer
          h(1)=o(1)*c(1); %hidden layer
          y(1)=h(1)*wy + by; %prediction
    
          for j=2:length(data)
            f(j)=1/(1+exp(-(h(j-1)*wf1 + data(j)*wf2 + bf)));%forget gate layer
            i(j)=1/(1+exp(-(h(j-1)*wi1 + data(j)*wi2 + bi)));%input gate layer
            ci(j)=tanh(h(j-1)*wc1 + data(j)*wc2 + bc);
            c(j)=f(j)*c(j-1) + i(j)*ci(j);%cell
            o(j)=1/(1+exp(-(h(j-1)*wo1 + data(j)*wo2 + bo))); %output gate layer
            h(j)=o(j)*c(j);%hidden layer
            y(j)=h(j)*wy + by;%prediction
          end
          cost=0;
          for j=1:train-seq
              cost=cost+1/(train-seq)*(y(seq+j-1)-data(seq+j))^2;
          end

          costchange(t)=cost;
      end
end

toc;

semilogy(costchange,'b-');

%out=[wf1,wf2,bf,wi1,wi2,bi,wc1,wc2,bc,wo1,wo2,bo,wy,by,c(end),h(end),y(end)]; %for comparison with srnnwb.m in comp_rnn_lsrm.m

%out=y(end); %for comp2_rnn_lstm.m

out=y(end)*maxdata;
end
          