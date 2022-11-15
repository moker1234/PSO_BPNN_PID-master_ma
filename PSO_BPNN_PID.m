clear all;
close all;

xite=0.50;
alfa=0.05; 
IN=4; H=5; Out=3;

ts=0.01;

% wi=0.50*rands(H,IN);
% wo=0.50*rands(Out,H);

% �������Ⱥ��ֵ�Ż��㷨��������Ȩ�س�ʼ������
wi=[
    2.4801   -0.9057    1.3533    3.2725
    1.4575   -1.0679    1.7680   -0.5159
    2.1270   -0.5822   -0.3259    3.8609
   -0.7626    0.5632   -0.6900   -1.7522
    2.1743    1.3040   -2.4301   -1.1352
   ];
wo=[
   -0.9053    1.4118    1.2013   -2.5795   -0.4790
    0.3864   -1.1645   -3.3277   -0.5947   -0.9984
    1.1042   -1.0132   -1.2603    2.3803   -1.1148

   ];

%[wi , wo]=PSO_InitW();
wi_init_save=wi;   wo_init_save=wo;   
wo_1=wo;  wo_2=wo; 
wi_1=wi;  wi_2=wi;
 
%M=[10,1,10];
%�����ϵ��
M=[9.9,9.8,9.4];

x=[0,0,0];
du_1=0;
u_1=0; u_2=0; u_3=0; u_4=0; u_5=0;u_6=0;u_7=0;
y_1=0; y_2=0; y_3=0; 
error_1=4; error_2=0;
  
Oh=zeros(H,1);
I=Oh;
u_lim=10;
  
sys=tf(523500,[1,87.35,10470]);
% sys=tf(400,[1,50,0]);
dsys=c2d(sys,ts,'z');
[num,den]=tfdata(dsys,'v');
num=[0,0.01704, 0.01443];den=[1.0,-1.6065,0.6065];
% [9.4, 1.81, 9.9];
kp_1=9.4;ki_1=1.81;kd_1=9.9;
for k=1:1:200 
    time(k)=k*ts;
    rin(k)=1.0; 
    
    error(k)=rin(k)-y_1;
    X(1)=error(k)-error_1;
    X(2)=error(k);
    X(3)=error(k)-2*error_1+error_2;
    xii=[X(1),X(2),X(3),1];
%     xii=[rin(k),y_1,error(k),1];
    xi=xii/norm(xii);
    epid=[X(1);X(2);X(3)];

    %%%ǰ�򴫲�----------------------------------------
    
    net2=xi*(wi');
     
    for j=1:1:H
        Oh(j)=( exp( net2(j)-exp(-net2(j)) ) )/(exp( net2(j)+exp(-net2(j)) ));

    end
   
    net3=wo*Oh;
      
    for l=1:1:Out
        K(l)=( exp( net3(l)-exp(-net3(l)) ) )/(exp( net3(l)+exp(-net3(l)) ));
        %K(l)=M*net3(l);
    end
%     dkp(k)=M(1)*K(1); dki(k)=M(2)*K(2); dkd(k)=M(3)*K(3);
    dkp(k)=K(1); dki(k)=K(2); dkd(k)=K(3);
    kp(k)=kp_1+dkp(k);ki(k)=ki_1+dki(k);kd(k)=kd_1+dkd(k);
    kp_1=kp(k);ki_1=ki(k);kd_1=kd(k);
    Kpid=[kp(k),ki(k),kd(k)];
    du(k)=Kpid*epid;
    
    if isnan(du(k))
        disp(net2);
    end
    u(k)=u_1+du(k); 
    if u(k)>u_lim
        u(k)=u_lim;
    end
    if u(k)<-u_lim
        u(k)=-u_lim;
    end
    
    yout(k)=-den(2)*y_1-den(3)*y_2+num(2)*u(k)+num(3)*u_1;
    
    %%%���򴫲�------------------------------------------------
    dyu(k)=sign((yout(k)-y_1)/(du(k)-du_1+0.0001));
    for j=1:1:Out
            dK(j)=4/(exp(net3(j))+exp(-net3(j)))^2;
           %dK(j)=M;
    end
      
    for l=1:1:Out
        delta3(l)=error(k)*dyu(k)*epid(l)*dK(l);
    end

    for l=1:1:Out
        for i=1:1:H
            d_wo=xite*delta3(l)*Oh(i)+alfa*(wo_1-wo_2);
        end
    end
%     d_wo=xite*delta3*Oh+alfa*(wo_1-wo_2);

    wo=wo_1+d_wo+alfa*(wo_1-wo_2);
    
    %����Mϵ��
%     for h = 1:Out
%         dM(h)=error(k)*K(h);
%         M(h)=M(h)+rite*dM(h);
%     end
    
    for i=1:1:H
        dO(i)=4/(exp(net2(i))+exp(-net2(i)))^2;
    end
    segma=delta3*wo;
    for i=1:1:H
        delta2(i)=dO(i)*segma(i);
    end
    d_wi=xite*delta2'*xi;
    wi=wi_1+d_wi+alfa*(wi_1-wi_2);
      
    wo_2=wo_1; wo_1=wo;
    wi_2=wi_1; wi_1=wi;
    du_1=du(k);
    
    u_7=u_6;u_6=u_5;u_5=u_4; u_4=u_3;u_3=u_2;u_2=u_1;u_1=u(k);   
    y_2=y_1; y_1=yout(k); 
    error_2=error_1; error_1=error(k);

end
figure(2);
[t,y]=BPNN_PID(0, 1);
plot(t,y,'g','Linewidth',2);
hold on;

% plot(time,rin,'r','Linewidth',2);
% xlabel('t/s');  ylabel('rin,yout');
% hold on ;

plot(time,yout,'c','Linewidth',2);
hold on;

[x,y]=classic_PID(0,1);
plot(x,y,'b','Linewidth',2);
legend('BPNN_PID_y','rin','y','classic_PID_y');
saveas(gcf,'myfig1.png')
%-----------------

% figure(3);
% plot(time,error,'r','Linewidth',2);
% xlabel('t/s');  ylabel('error');
% 
% figure(4);
% plot(time,u,'r','Linewidth',2);
% xlabel('t/s');  ylabel('u');
% 
% figure(5);
% subplot(311);
% plot(time,kp,'r','Linewidth',2);
% xlabel('t/s');  ylabel('kp');
% subplot(312);
% plot(time,ki,'g','Linewidth',2);
% xlabel('t/s');  ylabel('ki');
% subplot(313);
% plot(time,kd,'b','Linewidth',2);
% xlabel('t/s'); ylabel('kd');
% ykk=yout'*100;
% ukk=u';
% file_name='D:\Program Files\Polyspace\R2016aWorkplace\PSO_BPNN_PID-master_rare\Res_ParaMat.xlsx';  % ��Ҫ������ļ����Ƽ�·��
% xlswrite(file_name,ukk,'sheet1','A1') % B2 ��ʾд��excel�е�һ������λ��
% xlswrite(file_name,ykk,'sheet1','B1') 
