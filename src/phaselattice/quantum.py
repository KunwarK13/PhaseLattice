"""Circuit realizations of finite harmonic response models."""

import numpy as np


def _harmonic_circuit(coefficients, denominator, phase, selector_qubits):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate, StatePreparation, XGate
    from qiskit.quantum_info import SparsePauliOp

    amplitudes = np.zeros(2**selector_qubits)
    amplitudes[: len(coefficients)] = np.sqrt(np.abs(coefficients) / denominator)
    circuit = QuantumCircuit(selector_qubits + 1)
    circuit.append(StatePreparation(amplitudes), list(range(selector_qubits)))
    qubits = list(range(selector_qubits + 1))
    for j, coefficient in enumerate(coefficients):
        if coefficient:
            rotation = RYGate((j + 1) * phase).control(
                selector_qubits, ctrl_state=j, annotated=False
            )
            circuit.append(rotation, qubits)
            if coefficient < 0:
                circuit.append(
                    XGate().control(selector_qubits, ctrl_state=j, annotated=False), qubits
                )
    return circuit, SparsePauliOp("Z" + "I" * selector_qubits)


def circuit_for(coefficients, denominator):
    """Construct a circuit parameterized by a scalar angle in radians."""
    from qiskit.circuit import Parameter

    selector_qubits = max(1, int(np.ceil(np.log2(len(coefficients)))))
    return _harmonic_circuit(coefficients, denominator, Parameter("phase"), selector_qubits)


def reconstructed_circuit(coefficients, denominator, weights):
    """Construct a response-equivalent circuit parameterized by input features."""
    from qiskit.circuit import ParameterVector

    features = ParameterVector("x", len(weights))
    angle = 2 * np.pi * sum(float(weight) * feature for weight, feature in zip(weights, features))
    selector_qubits = max(1, int(np.ceil(np.log2(len(coefficients)))))
    circuit, observable = _harmonic_circuit(coefficients, denominator, angle, selector_qubits)
    return circuit, observable, features


def quantum_labels(coefficients, denominator, phases, shots=0, seed=0):
    """Evaluate statevector expectations or sample the response-qubit measurement."""
    from qiskit import ClassicalRegister
    from qiskit.primitives import StatevectorEstimator, StatevectorSampler

    phases = np.asarray(phases, dtype=float)
    if phases.ndim != 1 or not np.isfinite(phases).all():
        raise ValueError("Expected a finite vector of phase angles")
    if type(shots) is not int or shots < 0:
        raise ValueError("Shots must be a nonnegative integer")
    circuit, observable = circuit_for(coefficients, denominator)
    values = []
    if shots == 0:
        estimator = StatevectorEstimator()
        for start in range(0, len(phases), 128):
            batch = phases[start : start + 128, None]
            values.extend(
                estimator.run([(circuit, observable, batch)]).result()[0].data.evs.tolist()
            )
    else:
        measured = circuit.copy()
        measured.add_register(ClassicalRegister(1, "out"))
        measured.measure(circuit.num_qubits - 1, 0)
        # Qiskit resets integer seeds per bound circuit; one Generator preserves independent draws.
        sampler = StatevectorSampler(seed=np.random.default_rng(seed))
        for start in range(0, len(phases), 64):
            batch = phases[start : start + 64, None]
            draws = sampler.run([(measured, batch)], shots=shots).result()[0].data.out.array
            values.extend((1 - 2 * (draws[:, :, 0] & 1).mean(axis=1)).tolist())
    return np.asarray(values), dict(
        qubits=circuit.num_qubits,
        observable="Z on response qubit",
        shots_per_example=shots,
        total_shots=len(phases) * shots,
        circuit_ops=dict(circuit.count_ops()),
    )
