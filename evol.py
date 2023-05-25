import numpy.linalg as lg
import matplotlib.pyplot as plt
import numpy as np

class H_GL:
        def __init__(self, delta, n, alpha, lambda0,lambda_max):
            self.delta   = delta
            self.alpha   = alpha
            self.lambda0 = lambda0
            self.lambda_max = lambda_max
            self.n = n

        def Ht(self, t):
            n=self.n
            return np.array([[(-self.lambda0+self.alpha*t)**n ,self.delta*(-self.lambda0+self.alpha*t)], [  self.delta*(-self.lambda0+self.alpha*t), -(-self.lambda0+self.alpha*t)**n]])
 
        def H_lambda(self, lambda_):
            n=self.n
            return np.array([[lambda_**n ,self.delta*lambda_], [  self.delta*lambda_, -lambda_**n]])


        def Fidelity(self, lambda_):
            n=self.n
            return ( (self.n-1.0)**2.0 )* self.delta**2.0 * lambda_**(2.0*(n-2.0)) /  ( self.delta**2.0 + lambda_**(2.0*(n-1.0)) ) **2.0 / 4.0

        def E(self,lambda_):
            n = self.n
            return [-(lambda_**(2.0*self.n)+(lambda_**2.0)*self.delta**2.0)**0.5, (lambda_**(2*self.n)+(lambda_**2.0)*self.delta**2)**0.5]


        def Psi(self, lambda_):
            n = self.n
            E = self.E(lambda_)
            dummy = ((E[0]-lambda_)**2.0 + lambda_**(2*n) )**0.5
            return [lambda_**n, (E[0]-lambda_)]/dummy

        def Psi_(self, lambda_):
            e, psi = lg.eig( self.H_lambda( lambda_ ))
            return e, psi

        def Fidelity_fd(self, lambda_):
            step = 0.0001
            el , psil = self.Psi_( lambda_)
            er , psir = self.Psi_( lambda_+step)
       #    print (psil, ' ' ,psir,' ' ,np.dot(np.conj(psil),psir))
            dummy = (1.0-np.dot(np.conj(psil[0]),psir[0])**2.0)/step**2.0
            return dummy



class H_GP:
        def __init__(self, delta, n, alpha, lambda0,lambda_max):
            self.delta   = delta
            self.alpha   = alpha
            self.lambda0 = lambda0
            self.lambda_max = lambda_max
            self.n = n

        def Ht(self, t):
            n=self.n
            return np.array([[ (-self.lambda0+self.alpha*t)**n,self.delta,], [self.delta, -(-self.lambda0+self.alpha*t)**n]])

        def Fidelity(self, lambda_):
            n=self.n
            return (lambda_**(2*(n-1.0))*(self.delta**2.0)*n**2.0)/(self.delta**2.0 + lambda_**(2.0*n))**2.0/4.0

        def E(self,lambda_):
            n = self.n
            return [-(lambda_**(2.0*n) + self.delta**2.0)**0.5, (lambda_**(2.0*n) + self.delta**2.0)**0.5]

        def Psi(self, lambda_):
            n = self.n
            E = self.E(lambda_)
            dummy = ((E[0]-lambda_)**2.0 + lambda_**(2*n) )**0.5
            return [lambda_**n, (E[0]-lambda_)]/dummy


        def Fidelity_fd(self, lambda_):
            step = 0.0001
            psil = self.Psi( lambda_)
            psir = self.Psi( lambda_+step)
       #    print (psil, ' ' ,psir,' ' ,np.dot(np.conj(psil),psir))
            dummy = (1.0-np.dot(np.conj(psil),psir)**2.0)/step**2.0
            return dummy



class Crank_Nickelsson:
        def __init__(self, H, psi0, del_t, steps):
            self.H       = H 
            self.psi_0   = psi0
            self.psi_tmp = psi0
            self.psi_evolved = psi0
            self.del_t   = del_t
            self.steps   = steps
            self.I       = np.array([[1.,0],[0,1.]])
            self.prob_plus  = np.zeros(steps, dtype = float)
            self.prob_minus = np.zeros(steps, dtype = float)
            self.prob_trans = np.zeros(steps, dtype = float)
            self.e = np.zeros((steps,2), dtype = float)
            self.fidelity = np.zeros(steps, dtype = float)
            self.lambda_ = np.zeros(steps, dtype = float)
            self.fidelity_fd = np.zeros(steps, dtype = float)
            self.phase_plus  = 0+0j
            self.phase_minus = 0+0j

        def evol_operator(self, step):
            return np.dot( lg.inv( self.I + self.del_t*1j*0.5*self.H.Ht( (step+0.5)*self.del_t ) ), self.I - self.del_t*1j*0.5*self.H.Ht( (step+0.5)*self.del_t ) );
        
        def evolve(self):
            for i in range(self.steps):
                self.psi_evolved = np.dot ( self.evol_operator( i ), self.psi_evolved)
                e, psi = lg.eig( self.H.Ht( ( i +0.5)*self.del_t ) )

                if ( e[0] < e [1] ):
                  self.prob_plus[i]  =  (np.vdot( psi[:,1], self.psi_evolved ).real)**2.0 + (np.vdot( psi[:,1], self.psi_evolved ).imag)**2.0 
                  self.prob_minus[i] =  (np.vdot( psi[:,0], self.psi_evolved ).real)**2.0 + (np.vdot( psi[:,0], self.psi_evolved ).imag)**2.0
                  if (i==self.steps-1):
                     self.phase_plus = np.vdot( psi[:,1], self.psi_evolved )
                     self.phase_minus= np.vdot( psi[:,0], self.psi_evolved )
                else:
                  self.prob_minus[i] =  (np.vdot( psi[:,1], self.psi_evolved ).real)**2.0 + (np.vdot( psi[:,1], self.psi_evolved ).imag)**2.0
                  self.prob_plus[i]  =  (np.vdot( psi[:,0], self.psi_evolved ).real)**2.0 + (np.vdot( psi[:,0], self.psi_evolved ).imag)**2.0
                  if (i==self.steps-1):
                     self.phase_minus= np.vdot( psi[:,1], self.psi_evolved )
                     self.phase_plus = np.vdot( psi[:,0], self.psi_evolved )


                self.prob_trans[i]   =   self.psi_evolved[0].real**2.0 + self.psi_evolved[0].imag**2.0
                self.lambda_[i] = -self.H.lambda0+self.H.alpha*(i +0.5)*self.del_t  
                self.e[i] = self.H.E( self.lambda_[i] )
                self.fidelity[i] = self.H.Fidelity( self.lambda_[i] )
                self.fidelity_fd[i] = self.H.Fidelity_fd( self.lambda_[i] )
            if (e[0]<e[1]):
              print('final local wave function:',psi[:,0])
              print('final local wave function:',psi[:,1])
            else:
              print('final local wave function:',psi[:,1])
              print('final local wave function:',psi[:,0])
            return e, psi
# ---------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------#
def run_(alpha, delta, lambda0,lambda_max, sw_N, sw_GP_GL):
 
   if ( sw_GP_GL  =='GL'):
       H = H_GL(delta, sw_N, alpha, lambda0,lambda_max)
   if ( sw_GP_GL  =='GP'):
       H = H_GP(delta, sw_N, alpha, lambda0,lambda_max)
   preff  = sw_GP_GL + str(sw_N)  + '_del'+str(delta)  + '_alpha' + str(alpha) + '_lam0'+str(lambda0) + 'lammax' + str(lambda_max)
   print ( preff )
   H.delta=delta
   H.alpha=alpha
   fprobs = open('prob_' + preff + '.dat','w')
   fe_fidelity= open('efidelity_' + preff + '.dat','w')

   e, psi = lg.eig( H.Ht(0) )
   steps  = int(1.0*(H.lambda_max+H.lambda0)/(H.alpha*0.001))
   print("steps= ",steps, "  delta= ",delta)
   if (e[0] < e[1]):
        ck = Crank_Nickelsson(H, psi[:,0], 0.001, steps)
        print ('Initial wave function:',psi[:,0])
   else:
        ck = Crank_Nickelsson(H, psi[:,1], 0.001, steps)
        print  ('Initial wave function!:',psi[:,1])

   time = -H.lambda0 + H.alpha*ck.del_t*np.arange(ck.steps)
   ck.evolve()
   print('Evolved wavefunction:', ck.psi_evolved )

   e, psi = lg.eig( H.Ht(time[ck.steps-1]) )
   print ('Final instantaneus wavefunctions: ')
   print (e[0],psi[:,0])
   print (e[1],psi[:,1])
   
   for a in range(ck.steps):
         fe_fidelity.write(str(ck.lambda_[a])+' '+ str(ck.e[a][0]) + ' '+ str(ck.e[a][1]) + ' '+str(ck.fidelity[a])+ ' ' + str(ck.fidelity_fd[a])+'\n')
         fprobs.write(str(ck.lambda_[a])+' '+ str(ck.prob_trans[a])+ ' '+ str(ck.prob_minus[a]) +' '+ str(ck.prob_plus[a]) +'\n')
   
   return ck

# ---------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------#
def run_phase(alpha, delta, lambda0, sw_N, sw_GP_GL):
   lambda_max = 0.0
   if ( sw_GP_GL  =='GL'):
       H = H_GL(delta, sw_N, alpha, lambda0, lambda_max)
   if ( sw_GP_GL  =='GP'):
       H = H_GP(delta, sw_N, alpha, lambda0, lambda_max)
   preff  = sw_GP_GL + str(sw_N)  + '_del'+str(delta)  + '_alpha' + str(alpha) + '_lam0'+str(lambda0)
   print ( preff )
   H.delta=delta
   H.alpha=alpha
   fprobs = open('prob_' + preff + '.dat','w')
   fe_fidelity= open('efidelity_' + preff + '.dat','w')

   e, psi = lg.eig( H.Ht(0) )
   steps  = int(1.0*(H.lambda_max+H.lambda0)/(H.alpha*0.001))
   print("steps= ",steps, "  delta= ",delta)
   if (e[0] < e[1]):
        ck = Crank_Nickelsson(H, psi[:,0], 0.001, steps)
        print ('Initial wave function:',psi[:,0])
   else:
        ck = Crank_Nickelsson(H, psi[:,1], 0.001, steps)
        print  ('Initial wave function!:',psi[:,1])
   e, psi = ck.evolve()

   lambda_max = lambda0
   lambda0 = -0.001

   if ( sw_GP_GL  =='GL'):
       H = H_GL(delta, sw_N, alpha, lambda0, lambda_max)
   if ( sw_GP_GL  =='GP'):
       H = H_GP(delta, sw_N, alpha, lambda0, lambda_max)
   preff  = sw_GP_GL + str(sw_N)  + '_del'+str(delta)  + '_alpha' + str(alpha) + '_lam0'+str(lambda0)
   print ( preff )
   H.delta=delta
   H.alpha=alpha
#  fprobs = open('prob_' + preff + '.dat','w')
#  fe_fidelity= open('efidelity_' + preff + '.dat','w')

#   e, psi = lg.eig( H.Ht(0) )
   steps  = int(1.0*(H.lambda_max+H.lambda0)/(H.alpha*0.001))

   if (e[0] < e[1]):
        ck_minus = Crank_Nickelsson(H, psi[:,0], 0.001, steps)
        ck_plus = Crank_Nickelsson(H, psi[:,1], 0.001, steps)
        print ('Initial wave function:',psi[:,0])
        print ('Initial wave function:',psi[:,1])
   else:
        ck_plus = Crank_Nickelsson(H, psi[:,0], 0.001, steps)
        ck_minus= Crank_Nickelsson(H, psi[:,1], 0.001, steps)
        print ('Initial wave function:',psi[:,1])
        print ('Initial wave function:',psi[:,0])


#  time = -H.lambda0 + H.alpha*ck.del_t*np.arange(ck.steps)
   print('plus evolution')
   ck_plus.evolve()
   print('minus evolution')
   ck_minus.evolve()

   # phase calculation and prob calculation
   if (sw_GP_GL ==    'GP'):
     prob_trans = ck.phase_plus * ck_plus.phase_plus + ck.phase_minus * ck_minus.phase_plus
     phase1  = np.log( ck.phase_plus *  ck_plus.phase_plus ).imag
     phase2  = np.log( ck.phase_minus * ck_minus.phase_plus ).imag
   if (sw_GP_GL ==    'GL'):
     prob_trans = ck.phase_minus * ck_plus.phase_plus - ck.phase_plus * ck_minus.phase_plus
     phase1  = np.log( ck.phase_minus * ck_plus.phase_plus ).imag
     phase2  = np.log( ck.phase_plus * ck_minus.phase_plus ).imag


#  print ('Probs,phases, dphase: ',np.abs(prob_trans)**2, phase1,phase2, phase1-phase2)
#  print('amplitudes:', ck.phase_minus,ck.phase_plus , ck_plus.phase_plus,ck_minus.phase_plus )

#  print ('Phases:', phase, ck.ph1_plus, ck.ph1_minus, ck.ph2_plus, ck.ph2_minus )
#  print('Evolved wavefunction:', ck.psi_evolved )

#  e, psi = lg.eig( H.Ht(time[ck.steps-1]) )
#  print ('Final instantaneus wavefunctions: ')
#  print (e[0],psi[:,0])
#  print (e[1],psi[:,1])

   for a in range(ck.steps):
         fe_fidelity.write(str(ck.lambda_[a])+' '+ str(ck.e[a][0]) + ' '+ str(ck.e[a][1]) + ' '+str(ck.fidelity[a])+ ' ' + str(ck.fidelity_fd[a])+'\n')
         fprobs.write(str(ck.lambda_[a])+' '+ str(ck.prob_trans[a])+ ' '+ str(ck.prob_minus[a]) +' '+ str(ck.prob_plus[a]) +'\n')

   return ck, ck_minus, ck_plus

# ---------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------#
def run_series(deltarange, alpharange, sw_GP_GL, sw_N, lambda0,lambda_max):

   for alpha in alpharange:
     preff_vardelta = sw_GP_GL + str(sw_N) + '_vardelta' + '_alpha'+str(alpha) + '_lam0'+str(lambda0)+'lammax'+str(lambda_max)
     print ("----------- Var Delta --------------")
     ftranses_vardelta = open('trans'+preff_vardelta+'.dat','w')
     for delta in deltarange:
         preff_alpha = sw_GP_GL + str(sw_N) + '_varalpha' + '_delta'+str(delta) + '_lam0'+str(lambda0)
         ck = run_(alpha, delta, lambda0,lambda_max,sw_N, sw_GP_GL)
         steps = ck.steps
         ftranses_vardelta.write(str(delta)+' '+str(ck.prob_plus[steps-1])+' '+ str(ck.prob_minus[steps-1]) + ' ' +  str(ck.prob_trans[steps-1])+ '\n')
   return

def run_series_2(deltarange, alpharange, sw_GP_GL, sw_N, lambda0,lambda_max):

    for delta in deltarange:
     preff_varalpha = sw_GP_GL + str(sw_N) + '_varalpha' + '_delta'+str(delta) + '_lam0'+str(lambda0)
     print ("----------- Var Delta --------------")
     ftranses_vardelta = open('trans'+preff_varalpha+'.dat','w')
     for alpha in alpharange:
         ck = run_(alpha, delta, lambda0,lambda_max,sw_N, sw_GP_GL)
         steps = ck.steps
         ftranses_vardelta.write(str(alpha)+' '+str(ck.prob_plus[steps-1])+' '+ str(ck.prob_minus[steps-1]) + ' ' +  str(ck.prob_trans[steps-1]) +   '\n')

    return

def run_series_phase(deltarange, alpharange, sw_GP_GL, sw_N, lambda0):

   for alpha in alpharange:
     preff_vardelta = sw_GP_GL + str(sw_N) + '_vardelta_phase' + '_alpha'+str(alpha) + '_lam0'+str(lambda0)
     print ("----------- Var Delta --------------")
     ftranses_vardelta = open('trans'+preff_vardelta+'.dat','w')
     for delta in deltarange:
#        preff_alpha = sw_GP_GL + str(sw_N) + '_varalpha_phse' + '_delta'+str(delta) + '_lam0'+str(lambda0)
         ck, ck_minus, ck_plus = run_phase(alpha, delta, lambda0,sw_N, sw_GP_GL)
            # phase calculation and prob calculation
         if (sw_GP_GL ==    'GP'):
              prob_trans = ck.phase_plus * ck_plus.phase_plus + ck.phase_minus * ck_minus.phase_plus
              a1 = ck.phase_plus  * ck_plus.phase_plus 
              a2 = ck.phase_minus * ck_minus.phase_plus
              phase1  = np.log( ck.phase_plus *  ck_plus.phase_plus ).imag
              phase2  = np.log( ck.phase_minus * ck_minus.phase_plus ).imag
              phase3  = np.log( a1*a2.conjugate()).imag
         if (sw_GP_GL ==    'GL'):
              prob_trans = ck.phase_minus * ck_plus.phase_plus + ck.phase_plus * ck_minus.phase_plus
              phase1  = np.log( ck.phase_minus * ck_plus.phase_plus ).imag
              phase2  = np.log( ck.phase_plus * ck_minus.phase_plus ).imag
              a1 =  ck.phase_minus * ck_plus.phase_plus
              a2 =  ck.phase_plus * ck_minus.phase_plus
              phase3  = np.log( a1*a2.conjugate()).imag


         print ('Probs,phases, dphase: ',np.abs(prob_trans)**2, phase1,phase2, phase1-phase2, phase3)

       #  prob_trans = ck.phase_plus * ck_plus.phase_plus + ck.phase_minus * ck_minus.phase_plus
      #   phase1  = np.log( ck.phase_plus *  ck_plus.phase_plus ).imag
      #   phase2  = np.log( ck.phase_minus * ck_minus.phase_plus ).imag
         dphase = phase1-phase2
      #   print (np.abs(prob_trans)**2.0, phase1,phase2, phase1-phase2)

         ftranses_vardelta.write(str(delta)+' '+str(np.abs(prob_trans)**2.0)+' '+ ' ' + str(dphase) + ' ' +str(phase3) +  '\n')
   return



def fig1():
  alpha = 0.8
  delta=0.5
  sw_GP_GL='GL'
  lambda0=10.20
  for sw_N in range(1,5):
      run_(alpha, delta, lambda0,sw_N, sw_GP_GL  )

def fig2():
  alpha = 0.8
  delta=0.5
  sw_GP_GL='GP'
  lambda0=10.20
  for sw_N in range(1,5):
      run_(alpha, delta, lambda0,sw_N, sw_GP_GL )

def fig4_2():
   sw_GP_GL = 'GP'
   sw_N = 1
   lambda0=10.20
   deltarange = np.arange(0.01,2.01,0.1)
   alpharange = np.arange(0.2,1.1,0.4)
   run_series(deltarange, alpharange, sw_GP_GL, sw_N, lambda0)  
   run_series_2(deltarange, alpharange, sw_GP_GL, sw_N, lambda0)

   sw_GP_GL = 'GL'
   sw_N = 2
   lambda0=10.20
   deltarange = np.arange(0.01,2.01,0.1)
   alpharange = np.arange(0.2,1.1,0.4)
   run_series(deltarange, alpharange, sw_GP_GL, sw_N, lambda0)  
   run_series_2(deltarange, alpharange, sw_GP_GL, sw_N, lambda0)
   
def fig5_3  ():
  for sw_N in range(4,5):
    sw_GP_GL = 'GL'
#   sw_N = 2
    lambda0=10.0
    lambda_max=-0.01
    deltarange = np.arange(0.01,2.01, 0.01)
    alpharange = np.arange(0.1,1.4,0.2)
    run_series(deltarange, alpharange, sw_GP_GL, sw_N, lambda0,lambda_max)

    sw_GP_GL = 'GP'
#   sw_N = 2
    lambda0=10.0
    lambda_max=-0.01
    deltarange = np.arange(0.01,2.01, 0.01)
    alpharange = np.arange(0.1,1.4,0.2)
    run_series(deltarange, alpharange, sw_GP_GL, sw_N, lambda0, lambda_max)

def fig5_3_2():
  for sw_N in range(4,12,2):
#   sw_GP_GL = 'GP'
#   lambda0=10.0
#   lambda_max=10.2
#   deltarange = np.arange(0.01,2.01, 0.01)
#   alpharange = np.arange(0.1,1.2,0.2)
#   run_series_phase(deltarange, alpharange, sw_GP_GL, sw_N, lambda0)

    sw_GP_GL = 'GL'
    lambda0=10.0
    lambda_max=10.2
    deltarange = np.arange(0.01,2.01, 0.010)
    alpharange = np.arange(0.1,1.2,0.2)
    run_series_phase(deltarange, alpharange, sw_GP_GL, sw_N, lambda0)



fig5_3()


