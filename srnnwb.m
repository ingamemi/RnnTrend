function out = srnnwb(data,train,seq,times)
%srnn with bias bh and by
%from rnn3. extended the backpropagation terms
%train = training length
%seq = seq length
%times=adam algorithm iteration time

d=size(data);

if min(d)~=1||train >max(d)||seq >train
    disp('error : data needs to be a vector or incorrect testset/sequence length');
else
%    avg_data=mean(data);
%    var_data=mean((data-avg_data*ones(size(data))).^2);
%    data = (data-avg_data*ones(size(data)))./var_data;
%    data = data(1:train);
    %if max(data)>1 %put this part for srnnwtr.m - non-trend traiing part
       maxdata=max(abs(data));
       data = data(1:train)/maxdata; %cutting as training size and scaling data
    %end
    
% tic;
    h0= 0; %initial h0
    w1=1; %initial w1
    w2=1; %initial w2
    w3=1; %initial w3
    bh=0;
    by=0;
    
    h=zeros(size(data));
    y=zeros(size(data));
    h(1)=tanh(w1*h0 + w2*data(1)+bh); % with bias
    y(1)=w3*h(1)+by;   % with bias
    for i=2:length(data)
        h(i)=tanh(w1*h(i-1)+w2*data(i)+bh); %with bias
        y(i)=w3*h(i)+by;  % with bias
    end
   
    
    %Adam algorithm
    alpha=0.001; %stepsize
    beta1=0.9; %exponential decay rates for the moment estimates
    beta2=0.999; 
    mm=[0,0,0,0,0]; %initial 1st moment vector
    v=[0,0,0,0,0]; %initial 2nd moment vector
    t=0; %initialize timestep
    while t<times %try given times
          t=t+1;
          
          d1h=zeros(size(h));
          d2h=zeros(size(h));
          d1h(1)=0; %dh/dw1 = (1-h(1)^2)*h0 =0
          d2h(1)=(1-h(1)^2)*data(1);
          for i=2:length(h)
              d1h(i)=(1-h(i)^2)*(h(i-1)+w1*d1h(i-1)); %partial der. dh(i)/dw1
              d2h(i)=(1-h(i)^2)*(w1*d2h(i-1)+data(i)); %partial der. dh(i)/dw2
          end
          
          g(1)=0;g(2)=0;g(3)=0;g(4)=0;g(5)=0;
          m=train-seq;
          for j=1:m
              g(1)=g(1)+2/m*(y(seq+j-1)-data(seq+j))*w3*d1h(seq+j-1); % partial der. dcost/dw_1
              g(2)=g(2)+2/m*(y(seq+j-1)-data(seq+j))*w3*d2h(seq+j-1); % partial der. dcost/dw_2
              g(3)=g(3)+2/m*(y(seq+j-1)-data(seq+j))*h(seq+j-1);  % partial der. dcost/dw_3
              g(4)=g(4)+2/m*(y(seq+j-1)-data(seq+j))*w3*(1-h(seq+j-1)^2); %partial der. dcost/db_h
              g(5)=g(5)+2/m*(y(seq+j-1)-data(seq+j)); %partial der. dcost/db_y
          end
                  
 
          mm=beta1*mm+(1-beta1)*g;
          v=beta2*v+(1-beta2)*g.^2;
          mm_new=mm/(1-beta1^t);
          v_new=v/(1-beta2^t);
          
          w1=w1-alpha*mm_new(1)/(sqrt(v_new(1))+10^-8); %new w1,w2,w3,bh,by
          w2=w2-alpha*mm_new(2)/(sqrt(v_new(2))+10^-8);
          w3=w3-alpha*mm_new(3)/(sqrt(v_new(3))+10^-8);
          bh=bh-alpha*mm_new(4)/(sqrt(v_new(4))+10^-8);
          by=by-alpha*mm_new(5)/(sqrt(v_new(5))+10^-8);
          
          %check cost with new w1,w2,w3,bh
%           cost=0;
          h(1)=tanh(w1*h0 + w2*data(1)+bh);
          y(1)=w3*h(1)+by; 
          for i=2:length(data)
             h(i)=tanh(w1*h(i-1)+w2*data(i)+bh);
             y(i)=w3*h(i)+by;
          end
          cost=0;
          for i=1:train-seq
              cost=cost+1/(train-seq)*(y(seq+i-1)-data(seq+i))^2;
          end

          costchange(t)=cost;
      end
end

%toc;

%out= [w1,w2,w3];

% X=sprintf('%f,%f,%f,%f,%f,%f,%f ',w1,w2,w3,bh,by,h(end),costchange(end)); 
% disp(X)
  %out=[w1,w2,w3,costchange(end)];

 semilogy(costchange,'r-.')   

%out=[y(end),costchange]; % this is for testing rnn. need to command out when use srnnwtr.m  

%X=sprintf('%f',y(end)*maxdata);
%%X=sprintf('%f',y(end)*var_data+avg_data);
%display(X);

%out=[y(end)*var_data+avg_data,costchange(end)];



%out=[y(end),costchange(end)]; %changed the output to use this function in srnnwtr.m(trend training)

out=[y(end)*maxdata,costchange(end)]; % changed the output to use this function in srnnwtr.m(trend training)
 
%out=[w1,w2,w3,bh,by,h(end),y(end)]; %for comparison with slstm.m result in comp_rnn_lsrm.m

%out=y(end); %for comp2_rnn_lstm.m
end


