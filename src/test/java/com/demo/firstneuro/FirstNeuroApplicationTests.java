package com.demo.firstneuro;

import org.junit.Before;
import org.junit.Test;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.Assert.assertTrue;

@SpringBootTest
public class FirstNeuroApplicationTests {

    private ConnectedNeuron outputNeuron;
    private InputNeuron inputFriend;
    private InputNeuron inputVodka;
    private InputNeuron inputSunny;

    @Before
    public void setUp() {
        inputFriend = new InputNeuron();
        inputVodka = new InputNeuron();
        inputSunny = new InputNeuron();
    }

    @Test
    public void testExecutionAfterInvalidatingTheState() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.);

        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 1.);
        inputSunny.forwardSignalReceived(null, 1.);
        assert(outputNeuron.getForwardResult() > .7);

        inputFriend.forwardSignalReceived(null, 0.);
        inputVodka.forwardSignalReceived(null, 0.);
        inputSunny.forwardSignalReceived(null, 0.);
        assertTrue(outputNeuron.getForwardResult() < .3);
    }

    @Test
    public void testExecutionWhenResultShouldBeCloseTo1() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.);

        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 1.);
        inputSunny.forwardSignalReceived(null, 1.);
        assertTrue(outputNeuron.getForwardResult() > .7);
    }

    @Test
    public void testExecutionWhenResultShouldBeCloseTo0() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.);

        inputFriend.forwardSignalReceived(null, 0.);
        inputVodka.forwardSignalReceived(null, 0.);
        inputSunny.forwardSignalReceived(null, 0.);
        assertTrue(outputNeuron.getForwardResult() < .3);
    }

    @Test
    public void testExecutionWhenResultShouldBeMoreThen0dot5() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.);

        inputFriend.forwardSignalReceived(null, 1.);
        inputVodka.forwardSignalReceived(null, 0.);
        inputSunny.forwardSignalReceived(null, 1.);
        assertTrue(outputNeuron.getForwardResult() > .5);
    }

    @Test
    public void testExecutionWhenResultShouldBeLessThen0dot5() throws Exception {
        initiateOutputNeuronWithWeights(
                1.,
                1.,
                1.,
                -1.);

        inputFriend.forwardSignalReceived(null,0.);
        inputVodka.forwardSignalReceived(null,0.);
        inputSunny.forwardSignalReceived(null,1.);
        assertTrue(outputNeuron.getForwardResult() <= .5);
    }

    private void initiateOutputNeuronWithWeights(double wFriend,
                                                 double wVodka,
                                                 double wSunny,
                                                 double bias) {
        outputNeuron
                = new ConnectedNeuron.Builder()
                .bias(bias)
                .activationFunction(new StepFunction())
                .build();
        inputFriend.connect(outputNeuron, wFriend);
        inputVodka.connect(outputNeuron, wVodka);
        inputSunny.connect(outputNeuron, wSunny);
    }

}
