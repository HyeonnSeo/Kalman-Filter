#%%
import numpy as  np
import matplotlib.pyplot as plt

def drawPrediction(mu, sigma, ax):
    scale = 1e0

    theta = np.linspace(0,2*np.pi,100)
    p = np.vstack([np.cos(theta),np.sin(theta)])
    p = scale*sigma.dot(p)

    U,s,VT = np.linalg.svd(sigma)

    # ax.plot(mu[0],mu[1],'r.', markersize = 1)
    ax.plot(scale*s[0]*np.array([-U[0,0], U[0,0]]) + mu[0], scale*s[0]*np.array([-U[1,0], U[1,0]])+mu[1], 'r-', linewidth = 1)
    ax.plot(scale*s[1]*np.array([-U[0,1], U[0,1]]) + mu[0], scale*s[1]*np.array([-U[1,1], U[1,1]])+mu[1], 'r-', linewidth = 1)
    ax.plot(p[0,:]+mu[0], p[1,:]+mu[1],'r-',linewidth = 1)


def drawCorrection(mu, sigma,ax):
    scale = 1e0

    theta = np.linspace(0,2*np.pi,100)
    p = np.vstack([np.cos(theta),np.sin(theta)])
    p = scale*sigma.dot(p)

    U,s,VT = np.linalg.svd(sigma)

    # ax.plot(mu[0],mu[1],'b.', markersize = 1)
    ax.plot(scale*s[0]*np.array([-U[0,0], U[0,0]]) + mu[0], scale*s[0]*np.array([-U[1,0], U[1,0]])+mu[1], 'b-', linewidth = 1)
    ax.plot(scale*s[1]*np.array([-U[0,1], U[0,1]]) + mu[0], scale*s[1]*np.array([-U[1,1], U[1,1]])+mu[1], 'b-', linewidth = 1)
    ax.plot(p[0,:]+mu[0], p[1,:]+mu[1],'b-',linewidth = 1)



# Robot model parameter 
A = np.eye(2)
B = np.eye(2)
C = np.eye(2)

R = 1e0 * np.array([[1,1e-1],[1e-1,1]])     #로봇이 이동할때 노이즈의 covariance
Q = 1e0 * np.array([[1,0],[0,1]])           #센서 노이즈의 covariance


# Trajectory
radius = 1e1

# Initialize
# 로봇의 실제위치
x_true = np.zeros(2)

# Prediction 에 필요한 parameter
mu_pred = np.zeros(2)
# sigma_pred = np.random.randn(2,2)
sigma_pred = np.eye(2)

# Correction 에 필요한 param
mu_corr = np.zeros(2)
# sigma_corr = np.random.randn(2,2)
sigma_corr = np.eye(2)

 
# 그림 그리는데 필요한 값
rng = np.random.default_rng()

x_traj = np.full((1000,2), np.nan)
x_odom = np.full((1000,2), np.nan)
x_pred = np.full((1000,2), np.nan)
x_corr = np.full((1000,2), np.nan)

x_traj[0,:] = x_true.reshape(1,-1)
x_odom[0,:] = x_true.reshape(1,-1)
x_pred[0,:] = mu_pred.reshape(1,-1)
x_corr[0,:] = mu_corr.reshape(1,-1)
    
# Visualize
fig = plt.figure(1,figsize=(15,8))
ax = plt.subplot(111)
ax.axis("equal")

# 초기값 그리기
# ax.plot(z[0],z[1],'cx')
ax.plot(x_traj[:,0],x_traj[:,1],'ko-',linewidth=1)
ax.plot(x_odom[:,0],x_odom[:,1],color = np.array([0.5,0.5,0.5]),linewidth=1)

ax.set_xlim([-10,100])
ax.set_ylim([-10,50])

## 초기값
# drawPrediction(mu_pred, sigma_pred, ax)
# drawCorrection(mu_corr, sigma_corr, ax)

for t in range(1,12):
    ## measurement 생성
    # measurement Noise 를 정규분포에서 샘플링
    delta = rng.multivariate_normal(np.zeros(2), Q)     #평균이 0이고 공분산이 Q인 다변량 정규분포[N(0 ,1)]에서 무작위 샘플을 생성
    # print(delta)
    z = C.dot(x_true) + delta
    ax.plot(z[0],z[1],'cx')     # 측정값 그리기

    ## correction 진행 (mu_pred, sigma_pred, z)
    # Kalman Gain
    K = sigma_pred.dot(C.T).dot(np.linalg.inv(C.dot(sigma_pred).dot(C.T) + Q))

    mu_corr = mu_pred + K.dot(z-C.dot(mu_pred))
    sigma_corr = (np.eye(2)-K.dot(C)).dot(sigma_pred)

    x_corr[t,:] = mu_corr.reshape(1,-1)

    drawCorrection(mu_corr, sigma_corr, ax)

    ## control, 로봇 이동
    u = radius*np.array([np.cos(2*np.pi*t/100), np.sin(2*np.pi*t/100)]) # 원을 한바퀴 그리는 것을 100단계로 분리
    epsilon = rng.multivariate_normal(np.zeros(2), R)
    x_true = A.dot(x_true) + B.dot(u) + epsilon

    x_traj[t,:] = x_true.reshape(1,-1)
    ax.plot(x_traj[:,0],x_traj[:,1],'ko-',linewidth=1)

    x_odom[t,:] = x_odom[t-1,:] + u.reshape(1,-1)
    ax.plot(x_odom[:,0],x_odom[:,1],'o-',color = np.array([0.5,0.5,0.5]),linewidth=1)

    
    ## prediction
    mu_pred = A.dot(mu_corr) + B.dot(u)
    sigma_pred = A.dot(sigma_corr).dot(A.T) + R

    x_pred[t,:] = mu_pred.reshape(1,-1)
    drawPrediction(mu_pred, sigma_pred, ax)


#result



plt.figure(2, figsize=(15,5))
plt.plot(np.hypot( (x_odom[0:-1,0] - x_traj[0:-1,0]), (x_odom[0:-1,1]-x_traj[0:-1,1])),'ko-')
plt.plot(np.hypot( (x_pred[0:-1,0] - x_traj[0:-1,0]), (x_pred[0:-1,1]-x_traj[0:-1,1])),'ro-')
plt.plot(np.hypot( (x_corr[1:,0] - x_traj[0:-1,0]), (x_corr[1:,1]-x_traj[0:-1,1])),'bo-')
plt.legend(['odometry error', 'prediction error', 'correction error'])



plt.figure(3, figsize=(15,15))
plt.plot(x_traj[:,0], x_traj[:,1],'ko-')
plt.plot(x_odom[:,0], x_odom[:,1],'o-', color='gray')
plt.plot(x_pred[:,0], x_pred[:,1],'ro-')
plt.plot(x_corr[:,0], x_corr[:,1],'bo-')
plt.legend(['true', 'odometry', 'prediction', 'correction'])



# %%
