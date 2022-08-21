package com.demo.firstneuro;

public class StepFunction implements ActivationFunction{
    @Override
    public Double forward(Double x) {
        return x >= 1. ? 1. : 0.;
    }
}
