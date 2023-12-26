import numpy as np
from dimod import SimulatedAnnealingSampler, BinaryQuadraticModel
import dynex

class QCFD:
    def __init__(self):
        self.sampler = None
        self.bV = 8 # Number of Bits Variable (you can see it as precision 
                    # but don't increase it too much since we're going to normalize the solution later)

    def Lin2QUBO(self, matrix, vector):
        nV = len(vector) # Number of Variables
        bVs = nV * self.bV # The total number of binaries based on number of variables
        Q = np.zeros((bVs, bVs))
        A = np.dot(matrix.T, matrix)  # This is A^TA [See Documentation :D]
        b = vector  # This is b [See Documentation :D]
        for i in range(nV):
            for j in range(nV):
                for k in range(self.bV): # here we calculate the quadratic terms contribution
                    for l in range(self.bV):
                        idx_i = i*self.bV + k
                        idx_j = j*self.bV + l
                        Q[idx_i, idx_j] += A[i, j] * (2**k) * (2**l)  # Binary variable's contribution from the quadratic term
        for i in range(nV): # Now if we consider the linear terms contribution (-2b^T Ax)
            for k in range(self.bV):
                idx_i = i*self.bV + k
                # Based on (-2b^T A) [See Documentation :D]
                Q[idx_i, idx_i] -= 2 * b[i] * (2**k)  # index/interaction
        # [NOTE]: I used b^Tb to evaluate solution quality (doesn't affect Q)
        Q = {(i, j): Q[i][j] for i in range(len(Q)) for j in range(len(Q))}
        return Q
    
    def DecodeSol(self, sample, nV):
        QSol = np.zeros(nV) # just a placeholder for the solution
        for i in range(nV):
            bRep = [sample[i*self.bV + k] for k in range(self.bV)]
            xi = sum(val * 2**idx for idx, val in enumerate(bRep))
            QSol[i] = xi
        return QSol

    def compute(self, matrix, vector, compute='local'):
        matrix = np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix
        vector = np.array(vector) if not isinstance(vector, np.ndarray) else vector
        nV = len(vector)
        Q = self.Lin2QUBO(matrix, vector)
        if Q is None:
            raise ValueError("No QUBO model implemented!")
        sample = None
        if compute == 'dynex':
            # Dynex compute (DynexSDK)
            bqm = BinaryQuadraticModel.from_qubo(Q)
            model = dynex.BQM(bqm)
            sampler = dynex.DynexSampler(model, mainnet=True, description='DynexSDK QCFD Job')
            sampleset = sampler.sample(num_reads=1000, annealing_time = 200, debugging=False)
            sample = sampleset.first.sample
        elif compute == 'local':
            # Local compute (SA)
            self.sampler = SimulatedAnnealingSampler()
            sampleset = self.sampler.sample_qubo(Q)
            sample = sampleset.first.sample
        else:
            raise ValueError("Unsupported compute option!")
        solution = self.DecodeSol(sample, nV)
        print(solution)
        return solution