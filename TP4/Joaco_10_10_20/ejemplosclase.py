# -*- coding: utf-8 -*-

from numpy import linspace, logspace, diff, zeros
from numpy import cos, sin, exp, log, pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

####################################
# Implementación genérica de Taylor 2
# f(t,x): primera derivada de x respecto al tiempo
# g(t,x): segunda derivada de x respecto al tiempo
# x0: condición inicial
# t0, tf: tiempo inicial y final
# h: paso de integración
####################################
def taylor2(f,g,x0,t0,tf,h):
    N = int((tf-t0)/h)           #número de puntos
    t = linspace(t0,tf,N+1)
    n = x0.shape[0]         #dimensión del problema
    x = zeros((n,N+1))
    x[:,0] = x0
    i = h*h/2.0
    for k in range(N):
        x[:,k+1] = x[:,k]+h*f(t[k],x[:,k])+i*g(t[k],x[:,k])
        
    return t,x

####################################
# Implementación genérica de Heun
# f(t,x): derivada de x respecto al tiempo
# x0: condición inicial
# t0, tf: tiempo inicial y final
# h: paso de integración
####################################
def heun(f,x0,t0,tf,h):
    N = int((tf-t0)/h)          #número de puntos
    t = linspace(t0,tf,N+1)
    n = x0.shape[0]             #dimensión del problema
    x = zeros((n,N+1))
    x[:,0] = x0
    for k in range(N):
        f1 = h*f(t[k],x[:,k])
        f2 = h*f(t[k]+h,x[:,k]+f1)
        x[:,k+1] = x[:,k]+(f1+f2)/2.0
        
    return t,x

####################################
# Implementación genérica de Cauchy
# f(t,x): derivada de x respecto al tiempo
# x0: condición inicial
# t0, tf: tiempo inicial y final
# h: paso de integración
####################################
def cauchy(f,x0,t0,tf,h):
    N = int((tf-t0)/h)          #número de puntos
    t = linspace(t0,tf,N+1)
    n = x0.shape[0]             #dimensión del problema
    x = zeros((n,N+1))
    x[:,0] = x0
    h2 = h/2.0
    for k in range(N):
        x[:,k+1] = x[:,k]+h*f(t[k]+h2,x[:,k]+h2*f(t[k],x[:,k]))
        
    return t,x


########################
# EJEMPLO
########################
R = 1e3	            #Valor de la resistencia	
C = 1e-6	        #Valor de la capacidad
w = 2.0*pi*1000     #frecuencia angular de la señal de entrada
A = 1.0		        #amplitud de la señal de entrada
T = 5*2*pi/w	    #simulo cinco ciclos


####################################
# Solución
def xsol(t):
    x = -exp(-t/(R*C))+cos(w*t)+w*R*C*sin(w*t)
    x = (A/(1+(w*R*C)**2))*x
    return x

####################################
# Derivada primera de x
def dx(t,x):
    return ((A*cos(w*t)-x)/(R*C))

####################################
# Derivada segunda de x
def d2x(t,x):
    return ((-A*w*sin(w*t)-((A*cos(w*t)-x)/(R*C)))/(R*C))

####################################
# Plot ejemplo
def plotejemplo(h):
    x0 = zeros(1)
    t,xh = heun(dx,x0,0,T,h)
    t,xc = cauchy(dx,x0,0,T,h)
    t,xt = taylor2(dx,d2x,x0,0,T,h)
    x = xsol(t)
    fig, ax = plt.subplots()
    ax.plot(t, x, label='Solución')
    ax.plot(t, xh[0,:], label='Heun')
    ax.plot(t, xc[0,:], label='Cauchy')
    ax.plot(t, xt[0,:], label='Taylor 2')
    ax.legend()
    plt.title('Ejemplo')
    fig, ax = plt.subplots()
    ax.plot(t, xh[0,:]-x, label='Heun')
    ax.plot(t, xc[0,:]-x, label='Cauchy')
    ax.plot(t, xt[0,:]-x, label='Taylor 2')
    ax.legend()
    plt.title('Error')
    
####################################
# Errores ejemplo
def errorejemplo():
    n = 5
    N = logspace(1,5,n)
    h = T/N
    eh = zeros(n)
    ec = zeros(n)
    et = zeros(n)
    x0 = zeros(1)
    x = xsol(T)
    for k in range(n):
        t,xh = heun(dx,x0,0,T,h[k])
        t,xc = cauchy(dx,x0,0,T,h[k])
        t,xt = taylor2(dx,d2x,x0,0,T,h[k])
        eh[k] = abs(xh[0,-1]-x)
        ec[k] = abs(xc[0,-1]-x)
        et[k] = abs(xt[0,-1]-x)
    fig, ax = plt.subplots()
    ax.loglog(h, eh, label='Heun')
    ax.loglog(h, ec, label='Cauchy')
    ax.loglog(h, et, label='Taylor 2')
    # ax.loglog(h, (h/h[0])**2*eh[0], 'b--')
    # ax.loglog(h, (h/h[0])**2*ec[0], 'r--')
    # ax.loglog(h, (h/h[0])**2*et[0], 'g--')    
    ax.legend()
    plt.xlabel('h')
    plt.ylabel('error')
    plt.title('Error')
    print(diff(log(eh))/diff(log(h)))
    print(diff(log(ec))/diff(log(h)))
    print(diff(log(et))/diff(log(h)))
        
####################################
# Estimación error ejemplo
def esterrorejemplo(h):
    x0 = zeros(1)
    x = xsol(T)
    t,xh1 = heun(dx,x0,0,T,h)
    t,xc1 = cauchy(dx,x0,0,T,h)
    t,xt1 = taylor2(dx,d2x,x0,0,T,h)
    t,xh2 = heun(dx,x0,0,T,h/2)
    t,xc2 = cauchy(dx,x0,0,T,h/2)
    t,xt2 = taylor2(dx,d2x,x0,0,T,h/2)

    eh = abs(xh2[0,-1]-x)
    ec = abs(xc2[0,-1]-x)
    et = abs(xt2[0,-1]-x)
    eeh= abs(xh1[0,-1]-xh2[0,-1])/3.0
    eec= abs(xc1[0,-1]-xc2[0,-1])/3.0
    eet= abs(xt1[0,-1]-xt2[0,-1])/3.0
    print(abs(eh-eeh)/eh*100)
    print(abs(ec-eec)/ec*100)
    print(abs(et-eet)/et*100)
    return eh,eeh,ec,eec,et,eet

####################################
# Comparación con RK45
def comp45(h):
    x0 = zeros(1)
    t,xh = heun(dx,x0,0,T,h)
    # s45 = solve_ivp(dx,[0,T],x0,method='RK45', t_eval=None,
    #                 rtol = 1e-6, atol = 1e-8)
    # s45 = solve_ivp(dx,[0,T],x0,method='RK45', t_eval=None,
    #                 rtol = 1e-13, atol = 1e-14)
    s45 = solve_ivp(dx,[0,T],x0,method='RK45', t_eval=t,
                    rtol = 1e-13, atol = 1e-14)
    x = xsol(t)
    x1 = xsol(s45.t)

    fig, ax = plt.subplots()
    ax.plot(t, x, label='Solución')
    ax.plot(t, xh[0,:], label='Heun')
    ax.plot(s45.t, s45.y[0,:], label='RK45')
    ax.legend()
    plt.title('Ejemplo')
    fig, ax = plt.subplots()
    ax.plot(t[:-1],diff(t), label='Heun')
    ax.plot(s45.t[:-1],diff(s45.t), label='RK45')
    ax.legend()
    plt.title('Paso de integración')

    fig, ax = plt.subplots()
    ax.semilogy(t,abs(xh[0,:]-x), label='Heun')
    ax.semilogy(s45.t,abs(s45.y[0,:]-x1), label='RK45')
    ax.legend()
    plt.title('Error')

    print(s45.t.shape)
    print(t.shape)

plotejemplo(T/10000)
# plt.close('all')
# errorejemplo()
# esterrorejemplo(T/1e4)
#comp45(T/1000)
plt.show()
