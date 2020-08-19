#include "goma.h"

#include "mm_as.h"

#include "mm_as_structs.h"

#include "mm_fill_poisson.h"

#include <math.h>

int assemble_poisson(void)
{
 int eqn = R_POISSON;

 dbl g = -10.0 * exp(- (SQUARE(fv->x[0] - 0.5) - SQUARE(fv->x[1] - 0.5))/0.02);

 if (!pd->e[eqn])
   {
    return -1;
   }

 if (af->Assemble_Residual)
   {
    int peqn = upd->ep[eqn];
    for (int i = 0; i < ei->dof[eqn]; i++)
       {
        dbl diffusion = 0.0;

        if (pd->etm[eqn][LOG2_DIFFUSION])
          {
           for (int a = 0; a < pd->Num_Dim; a++)
              {
               diffusion -= bf[eqn]->grad_phi[i][a] * fv->grad_u[a];
              }
           diffusion *= fv->wt * bf[eqn]->detJ * fv->h3;
           diffusion *= pd->etm[eqn][LOG2_DIFFUSION];
          }

        dbl source = 0.0;
        if (pd->etm[eqn][LOG2_SOURCE])
          {
           source  = g * bf[eqn]->phi[i];
           source *= fv->wt * bf[eqn]->detJ * fv->h3;
           source *= pd->etm[eqn][LOG2_SOURCE];
          }

        lec->R[LEC_R_INDEX(peqn,i)] += diffusion + source;
       }
   }


 if (af->Assemble_Jacobian)
   {
    int peqn = upd->ep[eqn];
    for (int i = 0; i < ei->dof[eqn]; i++)
       {
        int var = POISSON;
        int pvar = upd->vp[var];
        for (int j = 0; j < ei->dof[var]; j++)
           {
            dbl diffusion = 0.0;

            if (pd->etm[eqn][LOG2_DIFFUSION])
              {
               for (int a = 0; a < pd->Num_Dim; a++)
                  {
                   diffusion -= bf[eqn]->grad_phi[i][a] * bf[var]->grad_phi[j][a];
                  }
               diffusion *= fv->wt * bf[eqn]->detJ * fv->h3;
               diffusion *= pd->etm[eqn][LOG2_DIFFUSION];
              }

            dbl source = 0.0;

            lec->J[LEC_J_INDEX(peqn,pvar,i,j)] += diffusion + source;
           }
       }
   }

 return 0;
}
