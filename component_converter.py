from perceval import StateVector, BasicState, Circuit, PostSelect, BSDistribution
from perceval.components import BS, PS, Processor
from math import *
import numpy as np
import qiskit as qk
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit.providers import Backend
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
import qiskit_ibm_runtime
from qiskit import transpile


def integer_to_binary(n):
    """
    translate an integer into the list of bits of its binary representation
    bits weight are in increasing order
    """
    l = []
    n_copy = n
    if n_copy == 0:
        return [0]
    while n_copy > 0:
        l.append(n_copy%2)
        n_copy //= 2
    return l


def binary_to_integer(l):
    s = 0
    for k in range(len(l)):
        s = 2*s + l[k]
    return s


def index_qbit_encoding(bs, n = None):
    """
    Use the Naive encoding to convert a FockState into its corresponding qbit representation
    :param bs: a BasicState
    :param n: The number of qbits for the encoding
    :return: an index in the qiskit qbit basis, and the number of qbits per modes
    """

    if not isinstance(bs, BasicState):
        raise TypeError (f"Only supports BasicState input, {type(bs)} was given")
    if n is None:
        if bs.n ==0:
            n = 1
        else:
            n = ceil(log2(bs.n))

    l = list(bs)
    idx = 0
    for k in range(bs.m):
        if l[k] >= 2**n:
            raise ValueError("The number of qbits is insufficient")
        else:
            idx += 2**(k*n)*l[k]
    if 2**n <= sum(l):
        raise Warning ("number of qbits might be insufficient with this number of photons")

    return idx


def qbit_encoding(sv, n):
    qbit_statevector = Statevector([0]*(2**(sv.m*n)))
    for bs, ampli in sv.items():
        qbit_statevector += ampli*Statevector.from_int(index_qbit_encoding(bs, n), 2**(bs.m*n))
    return qbit_statevector


def unitary_tensor_product(U,V):
    """
    makes some kind of matricial tensor product

    ex : H x H = 1/sqrt(2)( H  H )
                          ( H -H )

    :param U: np ndarray (matrix)
    :param V: np ndarray (matrix)
    :return: np ndarray (matrix)
    """
    n1,m1 = U.shape
    n2,m2 = V.shape
    n,m = n1*n2, m1*m2
    M = np.zeros((n,m), dtype = complex)
    for i in range(n1):
        for j in range(m1):
            M[i*n2:(i+1)*n2, j*m2:(j+1)*m2] = U[i,j]*V

    return M


def gate_based_phase_shifter(phi, n):
    """
    Gives the 2^n*2^n Unitary matrix corresponding to a phase shifter wer photon number is Naively encoded with qbits

    :param: phi is the corresponding Phase
    :param: n is the number of qbit, encoding at most 2^n-1photons
    """

    m = np.zeros((2**n,2**n), dtype=complex)
    for k in range(2**n):
        m[k,k] = np.exp(k*phi*1j)

    return m


def gate_based_beam_splitter(U, n ):
    """
    Gives the 2^2n*2^2n matrix corresponding to a beam splitter in Naive mode qbit encoding
    :param U: a 2x2 unitary representing a BeamSplitter
    :param n: The max number of qbits for the naive encoding of one mode
    :return: A 2^2n*2^2n matrix
    """
    M = np.zeros((2**(2*n), 2**(2*n)), dtype=complex)
    for in_index in range(2 ** (2 * n)):
        v = in_index // (2 ** n)  # |u,v> is the input basic state
        u = in_index%(2**n)
        nb_photon = u+v
        if nb_photon >= 2**n:
            M[in_index,in_index] = 1
        else:
            prefactor = factorial(u)*factorial(v)
            for s in range(nb_photon+1):
                t = nb_photon-s  # |s,t> is the output basic state
                denominator = factorial(s)*factorial(t)
                factor = sqrt(prefactor/denominator)
                out_index = s + t*(2**n)
                for k in range(u+1):
                    l = u-k
                    if k<=s and l<=t:
                        M[out_index, in_index] += factor*(comb(s, k)*comb(t, l)*
                                                (U[0,0]**k)*
                                                (U[0,1]**(s-k))*
                                                (U[1,0]**l)*
                                                (U[1,1]**(t-l)))
    return M


def input_precircuit(st: BasicState, nq_mode:int):
    n_qbit = st.m*nq_mode
    pre_circ = qk.QuantumCircuit(n_qbit)
    l = list(st)
    for mode in range(st.m):
        list_x = integer_to_binary(l[mode])
        for i, x in enumerate(list_x):
            if x == 1:
                pre_circ.x(mode*nq_mode + i)

    return pre_circ


def qbit_to_bs(number, qbit_mode, num_mode):

    bs_list = [0]*num_mode

    for k in range(num_mode):
        bs_list[k] = number % (2**qbit_mode)
        number //= 2**qbit_mode

    return BasicState(bs_list)


class NaiveBosonSamplingConverter:
    """
    Converter to build a gate based circuit equivalent to some boson sampling experiment

    Only support unitary components
    Only converts one and two modes components
    Does not support polarization / annotation
    Does not support symbolic parameters

    Only support sampling
    """
    def __init__(self, qbit_mode: int, backend=None, circuit=None):

        self.loqc = circuit
        self.qbit_mode = qbit_mode
        self.backend = backend
        self.min_detected_photon = None
        self.post_select = PostSelect()
        self.input_state = None

        if self.loqc is not None:
            self.build_gate_based_circuit()
        else:
            self.qiskit_circuit = None

    def set_optical_circuit(self, loqc: Circuit):

        if loqc.requires_polarization:
            raise NotImplementedError("can't convert circuits that requires polarization")

        if self.input_state is not None and loqc.m != self.input_state.m:
            raise ValueError("The number of modes of the input state is not compatible with the one of the circuit")

        for modes, component in loqc:
            if component.m > 2:
                raise NotImplementedError("can't convert components that acts on more than 2 modes")
        self.loqc = loqc

        self.build_gate_based_circuit()

    def build_gate_based_circuit(self):

        if self.input_state:
            self.qiskit_circuit = input_precircuit(self.input_state, self.qbit_mode)
        else:
            self.qiskit_circuit = QuantumCircuit(self.loqc.m*self.qbit_mode)

        for modes, component in self.loqc:

            if isinstance(component, PS):
                mode = modes[0]
                phi = x = float(component.get_parameters(all_params=True)[0])
                self.qiskit_circuit.unitary(gate_based_phase_shifter(phi, self.qbit_mode),
                                            [mode*self.qbit_mode + k for k in range(self.qbit_mode)])
            if component.m == 2:
                mode = modes[0]
                u_bs = np.array(component.U)
                self.qiskit_circuit.unitary(gate_based_beam_splitter(u_bs, self.qbit_mode),
                                            [mode*self.qbit_mode + k for k in range(2*self.qbit_mode)])

        self.qiskit_circuit.measure_all()

    def set_post_selection(self, ps: PostSelect):

        self.post_select = ps

    def set_photon_threshold(self, n_photon: int):
        self.min_detected_photon = n_photon

    def set_backend(self, backend):

        if not isinstance(backend, Backend):
            raise TypeError(f"backend must be a qiskit Backend, a {type(backend)} was given")

        self.backend = backend

    def set_input_state(self, fock_state: BasicState):

        if fock_state.n >= 2**self.qbit_mode:
            raise ValueError(f"you don't have enough qbit per modes to encode {fock_state.n} photons")

        if self.loqc is not None and fock_state.m != self.loqc.m:
            raise ValueError("The number of modes of the input state is not compatible with the one of the circuit")

        self.input_state = fock_state
        self.min_detected_photon = fock_state.n

        if self.loqc is not None:
            input_circuit = input_precircuit(fock_state, self.qbit_mode)
            self.qiskit_circuit = input_circuit.compose(self.qiskit_circuit)

    def sample(self, shots):

        if not isinstance(self.backend, AerSimulator):
            raise TypeError("Nooooooooooooooo!")

        sampler = AerSampler()
        job = sampler.run(self.qiskit_circuit, shots=shots, backend=self.backend)
        bsd = self._get_results(job)

        return bsd

    def _get_results(self, sampling_job):

        quasi_dists = sampling_job.result().quasi_dists[0]
        physical_perf = 1
        logical_perf = 1

        bsd = BSDistribution()

        for state, value in quasi_dists.items():

            bs = qbit_to_bs(state, self.qbit_mode, self.input_state.m)
            if bs.n < self.min_detected_photon:
                physical_perf -= value
            elif not self.post_select(bs):
                logical_perf -= value
            else:
                bsd[bs] += value

        bsd.normalize()

        return {"results": bsd, "logical_perf": logical_perf, "physical_perf": physical_perf}

    def sample_remote(self, shots):
        sampler = qiskit_ibm_runtime.Sampler(self.backend)
        job = sampler.run(self.qiskit_circuit, shots=shots)
        return job.job_id()

