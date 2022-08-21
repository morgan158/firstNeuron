package com.demo.firstneuro;

public class NeuroTestMain {
    public static void main(String[] args) {
        InputNeuron inputFriend = new InputNeuron();
        InputNeuron inputVodka = new InputNeuron();
        InputNeuron inputSunny = new InputNeuron();

        double bias = 0.;

        ConnectedNeuron outputNeuron =
                new ConnectedNeuron.Builder()
                        .bias(bias)
                        .activationFunction(new StepFunction())
                        .build();

        inputFriend.connect(outputNeuron, 0.5);
        inputVodka.connect(outputNeuron, 0.5);
        inputSunny.connect(outputNeuron, 0.5);

        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 1.);
        inputSunny.forwardSignalReceived(null, 0.);

        double result = outputNeuron.getForwardResult();
        System.out.printf("Prediction: %3f\n", result);
    }
}
