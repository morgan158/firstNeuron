package com.demo.firstneuro;

import javafx.util.Builder;

import java.util.*;

public class ConnectedNeuron implements Neuron{

    private final ActivationFunction activationFunction;
    private final Map<Neuron, Double> backwardConnections = new HashMap<>();
    private final Set<Neuron> forwardConnections = new HashSet<>();
    private final Map<Neuron, Double> inputSignals = new HashMap<>();
    private volatile int signalReceived;
    private final double bias;
    private volatile double forwardResult;

    private ConnectedNeuron(final ActivationFunction activationFunction,
                            final double bias) {
        this.activationFunction = activationFunction;
        this.bias = bias;
    }

    public double getForwardResult(){
        return forwardResult;
    }

    @Override
    public void forwardSignalReceived(final Neuron from, final Double value) {
        signalReceived++;
        inputSignals.put(from, value);
        if (backwardConnections.size() == signalReceived) {
            double forwardInputToActivationFunction =
                    backwardConnections
                            .entrySet()
                            .stream()
                            .mapToDouble(connection ->
                                    inputSignals.get(connection.getKey())
                            * connection.getValue())
                            .sum() + bias;

            double signalToSend
                    = activationFunction.forward(forwardInputToActivationFunction);
            forwardResult = signalToSend;

            forwardConnections
                    .stream()
                    .forEach(connection ->
                            connection
                                    .forwardSignalReceived(
                                            ConnectedNeuron.this,
                                            signalToSend
                                    ));

            signalReceived = 0;
        }
    }

    @Override
    public void addForwardConnection(Neuron neuron) {
        forwardConnections.add(neuron);
    }

    @Override
    public void addBackwardConnection(Neuron neuron, Double weight) {
        backwardConnections.put(neuron, weight);
        inputSignals.put(neuron, Double.NaN);
    }

    public static class Builder {
        private double bias = new Random().nextDouble();
        private ActivationFunction activationFunction;

        public Builder bias(final double bias) {
            this.bias = bias;
            return this;
        }

        public Builder activationFunction(final ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public ConnectedNeuron build() {
            if (activationFunction == null) {
                throw new RuntimeException("ActivationFunction need to be set in order to" +
                        " create a ConnectedNeuron");
            }
            return new ConnectedNeuron(
                    activationFunction,
                    bias);
        }
    }
}
