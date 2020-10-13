
# Original (working)
def ruku4(f, x0, t0, tf, h):
     step = h
     t_arr = []
     time_step = t0
     while time_step <= tf:
     t_arr.append(time_step)
     time_step += step


     x = np.zeros((len(t_arr), np.shape(x0)[0]))
     x[0,:] = x0

     stepOver2 = step / 2
     for k in range(1, len(t_arr)):
     f1	= f(t_arr[k-1], x[k-1,:])
     f2  = f(t_arr[k-1] + stepOver2, x[k-1,:] + stepOver2*f1)
     f3	= f(t_arr[k-1] + stepOver2, x[k-1,:] + stepOver2*f2)
     f4	= f(t_arr[k-1] + step, x[k-1,:] + step*f3)

     xnew = (step * (f1 + 2*f2 + 2*f3 + f4)) / 6
     x[k,:] = x[k-1,:] + xnew

     #print ("ruk4: Done.")
     return t_arr, x 
