function [wi_init, wo_init]=PSO_InitW()
%clear;
tic   %�ú�����ʾ��ʱ��ʼ
%���������
IN=4; H=5; Out=3;
%------������ʼ������----------------------------------------------
c1=2;    %1.4962;             %���ٳ���
c2=2 ;   %1.4962;             %���ٳ���
%w=0.7298;              %����Ȩ��
Wmax=0.9 ;   Wmin=0.4;  %׼����������Ȩ��˥����
MaxDT=200;            %����������
D=(H*IN)+(Out*H); %�����ռ�ά�������Ժ���sphere��δ֪��������
N=20;                      %��ʼ��Ⱥ�������Ŀ�� һ��20�����Ӿ��㹻��
Vmax=5;
Vmin=-5;
Pmax=5;
Pmin=-5;

%------��ʼ����Ⱥ�ĸ���(�����������޶�λ�ú��ٶȵķ�Χ)------------
for i=1:N
        x(i,:)=0.15*rands(1,D); 
        v(i,:)=0.15*rands(1,D); 
end

%------�ȼ���������ӵ���Ӧ�ȣ�����ʼ����������λ��y��ȫ������λ��Pg--------
for i=1:N
    p(i)=BPNN_Fitness(x(i,:)) ; %����ÿ��������Ӧ��
    y(i,:)=x(i,:);         %��ʼ����������λ��yΪ��ʱ�䲽t=0ʱ������λ��
end
Pg=x(1,:);              %PgΪȫ������λ�� �����ǳ�ʼ��
for i=2:N
    if BPNN_Fitness(x(i,:))<BPNN_Fitness(Pg)
        Pg=x(i,:);          %����ȫ������λ�� ��ʼ����� 
    end
end

%------������Ҫѭ�������չ�ʽ���ε�����ֱ�����㾫��Ҫ��------------
for t=1:MaxDT
    fprintf('��%d�ε���-----\n',t);
    %fprintf('��Ӧ��=%f\n',Pbest(t));
    for i=1:N
        w=Wmax-(t-1)*(Wmax-Wmin)/(MaxDT-1);
        v(i,:)=w*v(i,:)+c1*rand*(y(i,:)-x(i,:))+c2*rand*(Pg-x(i,:));
        v(i,find(v(i,:)>Vmax))=Vmax;    %���ܳ�������ٶ�
        v(i,find(v(i,:)<Vmin))=Vmin;      %���͹���С�ٶ�
        
        x(i,:)=x(i,:)+v(i,:);   %������ÿ�����ӵ�λ��
        x(i,find(x(i,:)>Pmax))=Pmax;
        x(i,find(x(i,:)<Pmin))= Pmin;
        if BPNN_Fitness(x(i,:))<p(i)
            p(i)=BPNN_Fitness(x(i,:));     %������Ӧ��
            y(i,:)=x(i,:);      %���¸������λ��
        end
        if p(i)<BPNN_Fitness(Pg)
            Pg=y(i,:);          %ÿһ�ε������������Ⱥ�����λ��
        end
    end
    Pbest(t)=BPNN_Fitness(Pg);     %����ÿһ����Ⱥ�������Ӧֵ
end
toc %�ú�����ʾ��ʱ����

%��þ�����Ⱥ�㷨�Ż���������Ȩֵ��ʼֵ
for t=1:H
       wi_init(t,:)=x(1,(t-1)*IN+1:t*IN);
end
for r=1:Out
       wo_init(r,:)=x(1, ( (IN*H+1)+(r-1)*H ): ( (IN*H+1)+r*H-1 ) );
end

%------������������--------
disp('*************************************************************')    
disp('������Ӧ��������λ��Ϊ��')
for i=1:D
    fprintf('x(%d)=%s\n',i,Pg(i));
end
fprintf('���õ����Ż���ֵΪ��%s\n',BPNN_Fitness(Pg));
disp('*************************************************************')
disp('�����������')
fprintf('����������%d\n',MaxDT);

figure(1);
plot(Pbest,'Linewidth',2);
title( ['��Ӧ������' ]);
grid on
xlabel('��������');ylabel('��Ӧ��');
wi_init
wo_init