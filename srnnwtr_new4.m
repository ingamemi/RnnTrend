function out=srnnwtr_new4(data,train,thrshld,times1,times2)

%non-trend part with max and min learning.

%new trend with 3 elements (duration,slope,epsilon) 
%this includes new trend finding method for srnnwtr.m

d=size(data);
tic;
if min(d)~=1||train >max(d)%||seq >train
    disp('error : data needs to be a vector or incorrect testset/sequence length');
else
    wholedata=data/max(abs(data)); %the next elements after train set %%added abs-10/18/18
    data = data(1:train)/max(abs(data)); %cutting as training size and scaling data
    
    datasize=size(data);
    if datasize(1)>1
        wholedata=wholedata';
        data=data';
    end
    
    idx=[1,length(data)]; %initial index - the first and the last
    maxidx=find(data==max(data)); 
    maxidx=maxidx(1); %initial index for maximum value  %note that it takes the first index if there are more than 2 position
    minidx=find(data==min(data)); 
    minidx=minidx(1); %initial index for minimum value 
    %thrshld=(maxidx-minidx)/sqrt(length(data)); %threshold is fixed with the first maxidx and minidx
    t=1;
    count(1)=1; %initial count for the while loop
    stopsign=1;
    idx=sort(unique([idx,maxidx,minidx]));
    mu(1)=data(1);
    mu(length(data))=data(end);
    while stopsign>0
        t=t+1;
        len=length(idx);
        addidx=[];
        for k=1:len-1
            for j=idx(k):idx(k+1)
                mu(j)=((data(idx(k+1))-data(idx(k)))/(idx(k+1)-idx(k)))*(j-idx(k))+data(idx(k));%linear line linking the min and max
            end
            after_mu=mu(idx(k):idx(k+1))-data(idx(k):idx(k+1)); %mu - data 
            
            maxidx=find(after_mu ==max(after_mu));
            maxidx=maxidx(1);
            minidx=find(after_mu==min(after_mu));
            minidx=minidx(1);
            if after_mu(maxidx)-after_mu(minidx) > thrshld
                addidx=[addidx,minidx+idx(k)-1,maxidx+idx(k)-1]; %if max-min between index is bigger than threshold, add those indices to this index set
            end
        end
        idx=sort(unique([idx,addidx]));
        count(t)=length(idx);
        stopsign=count(t)-count(t-1);
    end
    
    %out=length(idx);
    finalmu=zeros(1,length(idx)-1);
    trendmax=zeros(1,length(idx)-1);
    trendmin=zeros(1,length(idx)-1);
    for j=1:length(idx)-1
        x=(idx(j):idx(j+1));
        finalmu(x) = ((data(idx(j+1))-data(idx(j)))/(idx(j+1)-idx(j)))*(x-idx(j))+data(idx(j));%linear line linking the min and max
        eps=mean(data(x)-finalmu(x));
        finalmu(x) = finalmu(x)+eps;
        trendmax(j)=max(data(x)-finalmu(x));
        trendmin(j)=min(data(x)-finalmu(x));
    end
    
     
    %title('data and trend due to threshold');
    
    dur=zeros(length(idx)-1,1);
    slope=zeros(length(idx)-1,1);
    eps=zeros(length(idx)-1,1);
    tr=zeros(length(idx)-1,3);
    for i=1:length(idx)-1
        dur(i)=idx(i+1)-idx(i); %the duration l_i of tau_i 
        slope(i)=atan((data(idx(i+1))-data(idx(i)))/dur(i)); %the slope s_i of tau_i
        eps(i)=mean(data(idx(i):idx(i+1))-finalmu(idx(i):idx(i+1)));  %the epsilon eps_i of tau_i 
        tr(i,:)=[dur(i),slope(i),eps(i)]; %tau_i the trend
    end
    maxtr=max(tr(:,1)); %maximum trend length (duration)
    tr(:,1)=tr(:,1)/maxtr; %scaling the duration!!
    %training trend
    %-set sequence length as floor cut of the 1/7*(# of trend)
    trseq = max(1,floor(1/10*length(tr))); %trend sequence length
    trtrain = length(tr); %trend train data number
    %building initial ht's for trend training
    ht=zeros(length(trtrain),3); yt=zeros(size(ht));
    ht0=zeros(1,3); %initial h of trend
    wtx=ones(3,3);wtx(1,:)=0.01;wth=ones(3,3);wty=ones(3,3);bth=zeros(1,3);bty=zeros(1,3);%initial wt's. the initial condition for the first low of wtx as 0.1 to make it smaller than pi/2
    ht(1,:)= tanh(ht0*wth + tr(1,:)*wtx + bth);
    yt(1,:)= ht(1,:)*wty + bty;
    for i=2:trtrain
        ht(i,:)=tanh(ht(i-1,:)*wth + tr(i,:)*wtx + bth);
        yt(i,:)=ht(i,:)*wty +bty;
    end
    %Adam algorithm for trend part
    alpha=0.001; %stepsize
    beta1=0.9; %exponential decay rates for the moment estimates
    beta2=0.999; 
    mm=zeros(1,33); %initial 1st moment vector. 33 variables
    v=zeros(1,33); %initial 2nd moment vector
    t=0; %initialize timestep
    while t<times1 %try given times
          t=t+1;
          dhdwh=zeros(9,3,trtrain); %partial der. of h w.r.t.9 variables in W_h 
          dhdwt=zeros(9,3,trtrain); %partial der. of h w.r.t.9 variables in W_tr
          dhdbh=zeros(3,3,trtrain); %partial der. of h w.r.t.3 variables in b_h
          for k=1:9
                  mat=zeros(3,3); mat(k)=1;
                  dhdwh(k,:,1)= ([1,1,1]-(ht(1,:)).^2).*(ht0*mat); %initial par. der. dht/dw_h
                  dhdwt(k,:,1)= ([1,1,1]-(ht(1,:)).^2).*(tr(1,:)*mat); %initial par. der. dht/dw_tr
          end
          for k=1:3
              mat=zeros(1,3);mat(k)=1;
              dhdbh(k,:,1)= ([1,1,1]-(ht(1,:)).^2).*mat; %initial par.der. dht/db_h
          end
          
          for i=2:trtrain
              for k=1:9
                  mat=zeros(3,3); mat(k)=1;
                  dhdwh(k,:,i)= ([1,1,1]-(ht(i,:)).^2).*(dhdwh(k,:,i-1)*wth+ht(i-1,:)*mat); %par.der. dht/dw_h
                  dhdwt(k,:,i)= ([1,1,1]-(ht(i,:)).^2).*(dhdwt(k,:,i-1)*wth+tr(i,:)*mat); %par.der. dht/dw_tr
              end
              for k=1:3
                  mat=zeros(1,3);mat(k)=1;
                  dhdbh(k,:,i)= ([1,1,1]-(ht(i,:)).^2).*(dhdbh(k,:,i-1)*wth+mat); %par.der. dht/db_h
              end
          end
          
          g=zeros(1,33);
          m=trtrain-trseq;
          for k=1:9  % for W_h
              for j=1:m
                  A=ht(trseq+j-1,:)*wty+bty-tr(trseq+j,:); %cost=1/m sum_j( abs(A(j))^2 ). A: 1 by 3 vector
                  dA = dhdwh(k,:,trseq+j-1)*wty;   %dA(j)/d(vars in W_h)*W_y =dh(j)/d(vars in W_h)*W_y dA : 1 by 3 vector
                  g(k)=g(k)+2/m*A*dA';
              end
          end
          for k=10:18 % for W_tr
              for j=1:m
                  A=ht(trseq+j-1,:)*wty+bty-tr(trseq+j,:); %cost=1/m sum_j( abs(A(j))^2 ). A: 1 by 3 vector
                  dA = dhdwt(k-9,:,trseq+j-1)*wty;   %dA(j)/d(vars in W_tr)*W_y=dh(j)/d(vars in W_tr)*W_y dA : 1 by 3 vector
                  g(k)=g(k)+2/m*A*dA';
              end
          end
          for k=19:21  % for b_h
              for j=1:m
                  A=ht(trseq+j-1,:)*wty+bty-tr(trseq+j,:); %cost=1/m sum_j( abs(A(j))^2 ). A: 1 by 3 vector
                  dA = dhdbh(k-18,:,trseq+j-1)*wty;   %dA(j)/d(vars in b_h)*W_y=dh(j)/d(vars in b_h)*W_y dA : 1 by 3 vector
                  g(k)=g(k)+2/m*A*dA';
              end
          end
          for k=22:30 % for W_y
              for j=1:m
                  mat=zeros(3,3);mat(k-21)=1;
                  A=ht(trseq+j-1,:)*wty+bty-tr(trseq+j,:); %cost=1/m sum_j( abs(A(j))^2 ). A: 1 by 3 vector
                  dA = ht(trseq+j-1,:)*mat;   %dA(j)/d(vars in W_y)=h(j)*dW_y/d(vars in W_y) dA : 1 by 3 vector
                  g(k)=g(k)+2/m*A*dA';
              end
          end
          for k=31:33 % for b_y
              for j=1:m
                  mat=zeros(1,3);mat(k-30)=1;
                  A=ht(trseq+j-1,:)*wty+bty-tr(trseq+j,:); %cost=1/m sum_j( abs(A(j))^2 ). A: 1 by 3 vector
                  dA = mat;   % dA : 1 by 3 vector
                  g(k)=g(k)+2/m*A*dA';
              end
          end
          
          mm=beta1*mm+(1-beta1)*g;
          v=beta2*v+(1-beta2)*g.^2;
          mm_new=mm/(1-beta1^t);
          v_new=v/(1-beta2^t);
          
          mm_new_wh=zeros(3,3);mm_new_wh(1,:)=mm_new(1:3);mm_new_wh(2,:)=mm_new(4:6);mm_new_wh(3,:)=mm_new(7:9);
          mm_new_wt=zeros(3,3);mm_new_wt(1,:)=mm_new(10:12);mm_new_wt(2,:)=mm_new(13:15);mm_new_wt(3,:)=mm_new(16:18);
          mm_new_bh=mm_new(19:21);
          mm_new_wy=zeros(3,3);mm_new_wy(1,:)=mm_new(22:24);mm_new_wy(2,:)=mm_new(25:27);mm_new_wy(3,:)=mm_new(28:30);
          mm_new_by=mm_new(31:33);
          
          v_new_wh=zeros(3,3);v_new_wh(1,:)=v_new(1:3);v_new_wh(2,:)=v_new(4:6);v_new_wh(3,:)=v_new(7:9);
          v_new_wt=zeros(3,3);v_new_wt(1,:)=v_new(10:12);v_new_wt(2,:)=v_new(13:15);v_new_wt(3,:)=v_new(16:18);
          v_new_bh=v_new(19:21);
          v_new_wy=zeros(3,3);v_new_wy(1,:)=v_new(22:24);v_new_wy(2,:)=v_new(25:27);v_new_wy(3,:)=v_new(28:30);
          v_new_by=v_new(31:33);
          
          wth=wth-alpha*mm_new_wh./(sqrt(v_new_wh)+10^-8); %new wth,wtx,bth,wty,bty
          wtx=wtx-alpha*mm_new_wt./(sqrt(v_new_wt)+10^-8);
          wty=wty-alpha*mm_new_wy./(sqrt(v_new_wy)+10^-8);
          bth=bth-alpha*mm_new_bh./(sqrt(v_new_bh)+10^-8);
          bty=bty-alpha*mm_new_by./(sqrt(v_new_by)+10^-8);
          
          %check cost with new weights and biases
          ht(1,:)= tanh(ht0*wth + tr(1,:)*wtx + bth);
          yt(1,:)= ht(1,:)*wty + bty;
          for i=2:trtrain
              ht(i,:)=tanh(ht(i-1,:)*wth + tr(i,:)*wtx + bth);
              yt(i,:)=ht(i,:)*wty +bty;
          end
          totalcost=0;
          for j=1:m % m=trtrain-trseq
              A=ht(trseq+j-1,:)*wty+bty-tr(trseq+j,:);
              totalcost= totalcost + A*A';
          end
          
          costchange(t)=totalcost;
    end
    
    subplot(1,3,1)
    semilogy(costchange,'b-');   %training error of trend part
    title('trend training error');  
    
    
    % start non-trend part training. - with min and max of each trend
    % xdata=data-finalmu;  % data- trend
     ntr=length(trendmax); %non-trend length = trend size
     ntrseq=max(1,floor(ntr/10)); %sequence length for non-trend trainin 
     prdtr_len= floor(maxtr*yt(end,1)) ; % flooring the predicted trend length.
     ntrres=systemrnn3(trendmax,trendmin,ntr,ntrseq,times2); %nontrend learning result
     ntrmax=ntrres(1);
     ntrmin=ntrres(2);
     
     subplot(1,3,2)
     lastelt=idx(end)+prdtr_len;
      plot(idx(end):lastelt,wholedata(idx(end):lastelt),'k:'); %whole data for the prediction part
      hold on
      plot(idx(end):lastelt,tan(yt(end,2)).*(idx(end):lastelt)-tan(yt(end,2))*idx(end)+data(idx(end))+yt(end,3),'r-') %show prediction
      hold on
      plot(idx(end):lastelt,tan(yt(end,2)).*(idx(end):lastelt)-tan(yt(end,2))*idx(end)+data(idx(end))+yt(end,3)+ntrmax,'.-g'); %show final prediction
      hold on
      plot(idx(end):lastelt,tan(yt(end,2)).*(idx(end):lastelt)-tan(yt(end,2))*idx(end)+data(idx(end))+yt(end,3)+ntrmin,'.-g'); 
  
     X=sprintf('prediction of nontrendvalue =(%f,%f)',ntrmax,ntrmin);
     disp(X)
     
     subplot(1,3,3)
     lastelt=idx(end)+prdtr_len;
     lastpt=min(lastelt+2*prdtr_len,length(wholedata));
    plot(idx(end-3):lastpt,wholedata(idx(end-3):lastpt),':k'); %whole data for the prediction part
    hold on
    plot(idx(end-3):idx(end), mu(idx(end-3):idx(end)),'.-b'); % show exsting (2) treds
    hold on
     plot(idx(end):lastelt,tan(yt(end,2)).*(idx(end):lastelt)-tan(yt(end,2))*idx(end)+data(idx(end))+yt(end,3),'r-') %show prediction
      hold on
      plot(idx(end):lastelt,tan(yt(end,2)).*(idx(end):lastelt)-tan(yt(end,2))*idx(end)+data(idx(end))+yt(end,3)+ntrmax,'.-g'); %show final prediction
      hold on
      plot(idx(end):lastelt,tan(yt(end,2)).*(idx(end):lastelt)-tan(yt(end,2))*idx(end)+data(idx(end))+yt(end,3)+ntrmin,'.-g'); 
    
         

     
     
%      res=zeros(1,prdtr_len);
%      for i=1:prdtr_len % yt(end,1) = the predicted trend 
%          result =srnnwb(xdata,length(xdata),floor(length(xdata)/length(tr)),times2); %give sequence length as ((# of data)/(#of trends)
%          res(i)=result(1); % result(y_t) of simple rnn of non-trend part
%          xdata=[xdata,res(i)]; %extend input data for srnn with predicted result
%      end
    
%     trendresultvalue=tan(yt(end,2)).*(idx(end):idx(end)+prdtr_len-1)-tan(yt(end,2))*idx(end)+data(idx(end))+yt(end,3); %predicted trend values
%      Y= trendresultvalue + res; %combining trend result and non-trend result
   
    
    
    
    
%     X=sprintf('predict trend : (duration,slope,epsilon)=(%f,%f,%f)',maxtr*yt(end,1),yt(end,2),yt(end,3));
%     disp(X)
%     X=sprintf('(#trend,sequence length for trend)=(%f,%f)',length(tr),trseq);
%     disp(X)
%     X=sprintf('non-trend final error = %f',result(2));
%     disp(X)

%     out=totalcost;
%     subplot(2,2,2)
%     semilogy(costchange,'b-');   %training error of trend part
%     title('trend training error');      
    
%     subplot(2,2,3)
%     plot((train-3*floor(maxtr*yt(end-1,1)):min(length(wholedata),train+2*prdtr_len)),wholedata(train-3*floor(maxtr*yt(end-1,1)):min(length(wholedata),train+2*prdtr_len)),':k')
%     hold on
%     plot((train-3*floor(maxtr*yt(end-1,1)):train),finalmu(train-3*floor(maxtr*yt(end-1,1)):train),'r')
%     hold on
%     plot((train:train+prdtr_len-1),trendresultvalue,'b')
%     hold on
%     plot((train:train+prdtr_len-1),Y,'m+')
    
%     subplot(2,2,4)
%     plot((train:train+prdtr_len-1),wholedata(train:train+prdtr_len-1),':k')
%     hold on
%     plot((train:train+prdtr_len-1),trendresultvalue,'b')
%     hold on
%     plot((train:train+prdtr_len-1),Y,'m+')
    
    %X=sprintf('train-3*floor(maxtr*yt(end-1,1)) and train = %f,%f',train-3*floor(maxtr*yt(end-1,1)), train);
    %disp(X)
 
    
toc;
end       
            
            
        
        
        
    
    