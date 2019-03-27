"use-strict";

(function($) {

    // Tensor Object
    class TensorObject {
        constructor() {
            // Create tensor object
            this.Shape = [];
            this.Values = [];
        }

        // Print
        Print() {
            // Check shape
            switch (this.Shape.length) {
                case 0: // Scalar
                    console.log(`${this.Values} \nScalar \n`);
                    break;
                case 2: // Matrix
                    let printLog = ``;
                    for (let i = 0; i < this.Shape[0]; i++) {
                        printLog = printLog + `[ `;
                        for (let j = 0; j < this.Shape[1]; j++) {
                            printLog = printLog + ` ${this.Values[i][j]} `;
                        }
                        printLog = printLog + ` ] \n`;
                    }
                    console.log(printLog + `Matrix : (${this.Shape[0]}x${this.Shape[1]}) \n`);
                    break;
            }
        }
        // Random
        Random(min, max, floor) {
            let rts = this.Copy();
            // Check shape
            switch (this.Shape.length) {
                case 0: // Scalar
                    rts.Values = Random(min, max, floor)
                case 2: // Matrix
                    // Assign random values
                    for (let i = 0; i < this.Shape[0]; i++) {
                        for (let j = 0; j < this.Shape[1]; j++) {
                            rts.Values[i][j] = Random(min, max, floor);
                        }
                    }
            }
            return rts;
        }

        // Copy tensor
        Copy() {
            let rts = new TensorObject();
            rts.Shape = JSON.parse(JSON.stringify(this.Shape));
            rts.Values = JSON.parse(JSON.stringify(this.Values));
            return rts;
        }

        // Add
        Add(arg) {
            return elementWise(this, arg, 0);
        }
        // Sub
        Sub(arg) {
            return elementWise(this, arg, 1);
        }
        // Mul
        Mul(arg) {
            return elementWise(this, arg, 2);
        }
        // Square
        Square() {
            return elementWise(this, this, 2);
        }
        // Div
        Div(arg) {
            return elementWise(this, arg, 3);
        }
        // Add all
        AddAll() {
            let sum = 0;
            // Check shape
            switch (this.Shape.length) {
                case 0: // Scalar
                    return this.Values;
                    break;
                case 2: // Matrix
                    for (let i = 0; i < this.Shape[0]; i++) {
                        for (let j = 0; j < this.Shape[1]; j++) {
                            sum += this.Values[i][j];
                        }
                    }
            }
            return sum;
        }

        // Map function
        Map(callback) {
            let rts = this.Copy();
            // Checks for valid callback function
            if (callback != undefined && typeof callback != "function") {
                return rts;
            } else if (callback == undefined) {
                return rts;
            }
            // Check shape
            switch (this.Shape.length) {
                case 0: // Scalar
                    rts.Values = callback(rts.Values);
                    break;
                case 2: // Matrix
                    for (let i = 0; i < this.Shape[0]; i++) {
                        for (let j = 0; j < this.Shape[1]; j++) {
                            rts.Values[i][j] = callback(rts.Values[i][j]);
                        }
                    }
            }
            return rts;
        }

        // Transpose 
        Transpose() {
            let rts = this.Copy();
            // Check shape
            switch (this.Shape.length) {
                case 0: // Scalar 
                    return rts;
                    break;
                case 2: // Matrix
                    rts.Shape = [this.Shape[1], this.Shape[0]];
                    rts.Values = [];
                    for (let i = 0; i < rts.Shape[0]; i++) {
                        let r = [];
                        for (let j = 0; j < rts.Shape[1]; j++) {
                            r.push(this.Values[j][i]);
                        }
                        rts.Values.push(r);
                    }
            }
            return rts;
        }

        // Add cols
        AddCols() {
            let rts = new TensorObject();
            // Check shape
            switch (this.Shape.length) {
                case 0: // Scalar 
                    return this.Copy();
                    break;
                case 2: // Matrix
                    rts.Shape = [this.Shape[0], 1];
                    rts.Values = [];
                    for (let i = 0; i < this.Shape[0]; i++) {
                        let sum = 0;
                        for (let j = 0; j < this.Shape[1]; j++) {
                            sum += this.Values[i][j];
                        }
                        rts.Values.push([sum]);
                    }
            }
            return rts;
        }
        // Col extend
        ColExtend(scale) {
            let rts = this.Copy();
            // Check shape
            switch (this.Shape.length) {
                case 0: // Scalar
                    return this;
                case 2: // Matrix
                    rts.Shape = [rts.Shape[0], rts.Shape[1] * scale];
                    rts.Values = [];
                    for (let i = 0; i < rts.Shape[0]; i++) {
                        let r = [];
                        for (let j = 0; j < rts.Shape[1]; j++) {
                            r.push(this.Values[i][parseInt(j / scale)]);
                        }
                        rts.Values.push(r);
                    }
            }
            return rts;
        }
        // Make batches
        MakeBatches(size) {
            let rts = [];
            // Check shape
            switch (this.Shape.length) {
                case 0: // Scalar
                    return [this.Copy()];
                case 2: // Matrix
                    let totalBatches = this.Shape[1] / size;
                    if (totalBatches * size != this.Shape[1]) {
                        totalBatches += 1;
                    }
                    let initial = size;
                    for (let t = 0; t < totalBatches; t++) {
                        initial = t * size;
                        let limit = initial + size;
                        if (limit > this.Shape[1]) { // More than columns
                            limit = this.Shape[1];
                        }
                        let nts = Tensor([this.Shape[0], limit - initial]);
                        nts.Values = [];
                        for (let i = 0; i < this.Shape[0]; i++) {
                            let r = [];
                            for (let j = initial; j < limit; j++) {
                                r.push(this.Values[i][j]);
                            }
                            nts.Values.push(r);
                        }
                        rts.push(nts);
                    }
                    break;
                default:
                    return [this.Copy()];
            }
            return rts;
        }
    }

    // Tensor
    function Tensor(shape, values) {
        let ts = new TensorObject();
        let create = false;
        if (shape == undefined) {
            // No shape provided
            return ts;
        }
        if (values == undefined) {
            create = true;
        } else {
            // Tensor shape
            switch (shape.length) {
                case 0: // Scalar
                    if (typeof values != "number") {
                        create = true;
                    } else { ts.Values = values; }
                    break;
                case 1: // Vector
                    if (values.constructor != Array) {
                        create = true;
                    } else if (values.constructor == Array) {
                        // Check each element
                        for (let i = 0; i < values.length; i++) {
                            if (typeof values[i] != "number") { create = true; break; }
                        }
                        if (create == false) {
                            ts.Values = toMatrix(values);
                            ts.Shape = ts.Shape = [shape[0], 1];
                        }
                    }
                    break;
                case 2: // Matrix
                    if (values.constructor != Array) {
                        create = true;
                    } else if (values.constructor == Array) {
                        // Check each element
                        for (let i = 0; i < values.length; i++) {
                            if (values[i].constructor != Array) { create = true; break; } else if (values[i].constructor == Array) {
                                for (let j = 0; j < values[i].length; j++) {
                                    if (typeof values[i][j] != "number") {
                                        create = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if (create == false) {
                            ts.Values = values;
                            ts.Shape = shape;
                        }
                    }
                    break;
                default:
                    // Not supported
                    return ts;
            }
        }
        // Invalid values provides
        if (create == true) {
            // Create with value
            switch (shape.length) {
                case 0: // Scalar
                    ts.Values = 0.0;
                    break;
                case 1: // Vector
                    let vector = [];
                    for (let i = 0; i < shape[0]; i++) {
                        vector.push(0.0);
                    }
                    ts.Values = toMatrix(vector.slice());
                    ts.Shape = [shape[0], 1];
                    break;
                case 2: // Matrix
                    let matrix = [];
                    for (let i = 0; i < shape[0]; i++) {
                        let row = [];
                        for (let j = 0; j < shape[1]; j++) {
                            row.push(0.0);
                        }
                        matrix.push(row);
                    }
                    ts.Values = matrix.slice();
                    ts.Shape = shape;
            }
        }
        return ts;
    }

    // Random function
    function Random(min, max, floor) {
        // Checks if min or max are given
        if (min == undefined || typeof min != "number") {
            min = -1;
        }
        if (max == undefined || typeof min != "number") {
            max = 1;
        }
        let randNumb = Math.random();
        if (min < 0) {
            randNumb *= (max + Math.abs(min));
        } else {
            randNumb *= (max - min);
        }
        randNumb += min;
        if (floor == true) {
            randNumb = Math.floor(randNumb);
            if (min < 0 && randNumb < 0) {
                randNumb += 1;
            }
            if (min == max) {
                return min;
            }
        }
        return randNumb;
    }
    // Sigmoid function
    function Sigmoid(x) {
        // If x is not a number
        if (typeof x != "number") {
            return 1;
        }
        return 1 / (1 + Math.exp(-x));
    }
    // Differential of sigmoid
    function Dsigmoid(y) {
        // If y is not a number
        if (typeof y != "number") {
            return 1;
        }
        return y * (1 - y);
    }

    // Relu function
    function Relu(x) {
        // If x is not a number
        if (typeof x != "number") {
            return 1;
        }
        return Math.max(0, x);
    }

    // Differentiation of relu
    function Drelu(y) {
        // If y is not a number
        if (typeof y != "number") {
            return 1;
        }
        if (y <= 0) {
            return 0;
        } else {
            return 1;
        }
    }

    // Mean square error
    function meanSquareError(predict, output, batchSize) {
        let rts = new TensorObject();
        rts.Shape = [];
        rts.Values = predict.Sub(output).Square().AddAll() / batchSize.Values;
        return rts;
    }

    // Convert to matrix
    function toMatrix(vector) {
        let matrix = [];
        for (let i = 0; i < vector.length; i++) {
            let row = [];
            row.push(vector[i]);
            matrix.push(row);
        }
        return matrix;
    }

    // Element wise operation
    function elementWise(ts, arg, opt) {
        let rts = ts.Copy();
        // Check shape
        switch (rts.Shape.length) {
            case 0: // Scalar
                // Check arg
                switch (arg.Shape.length) {
                    case 0: // Scalar
                        switch (opt) {
                            case 0:
                                rts.Values += arg.Values;
                                break;
                            case 1:
                                rts.Values -= arg.Values;
                                break;
                            case 2:
                                rts.Values *= arg.Values;
                                break;
                            case 3:
                                rts.Values /= arg.Values;
                                break;
                        }
                        break;
                    case 2: // Matrix
                        rts.Shape = arg.Shape.slice();
                        let matrix = [];
                        for (let i = 0; i < rts.Shape[0]; i++) {
                            let r = [];
                            for (let j = 0; j < rts.Shape[1]; j++) {
                                switch (opt) {
                                    case 0:
                                        r.push(rts.Values + arg.Values[i][j]);
                                        break;
                                    case 1:
                                        r.push(rts.Values - arg.Values[i][j]);
                                        break;
                                    case 2:
                                        r.push(rts.Values * arg.Values[i][j]);
                                        break;
                                    case 3:
                                        r.push(rts.Values / arg.Values[i][j]);
                                        break;
                                }
                            }
                            matrix.push(r);
                        }
                        rts.Values = matrix.slice();
                        break;
                }
                break;
            case 2: // Matrix
                // check arg
                switch (arg.Shape.length) {
                    case 0: // Scalar
                        for (let i = 0; i < rts.Shape[0]; i++) {
                            for (let j = 0; j < rts.Shape[1]; j++) {
                                switch (opt) {
                                    case 0:
                                        rts.Values[i][j] += arg.Values;
                                        break;
                                    case 1:
                                        rts.Values[i][j] -= arg.Values;
                                        break;
                                    case 2:
                                        rts.Values[i][j] *= arg.Values;
                                        break;
                                    case 3:
                                        rts.Values[i][j] /= arg.Values;
                                        break;
                                }
                            }
                        }
                        break;
                    case 2: // Matrix
                        // Check for multiplication
                        if (rts.Shape[1] == arg.Shape[0] && opt == 2) { // Multiply matrix
                            rts.Shape = [rts.Shape[0], arg.Shape[1]];
                            let matrix = [];
                            for (let i = 0; i < rts.Shape[0]; i++) {
                                let r = [];
                                for (let j = 0; j < rts.Shape[1]; j++) {
                                    let sum = 0;
                                    for (let r = 0; r < arg.Shape[0]; r++) {
                                        sum += rts.Values[i][r] * arg.Values[r][j];
                                    }
                                    r.push(sum);
                                }
                                matrix.push(r);
                            }
                            rts.Values = matrix;
                            return rts;
                        }
                        // Check dimensions for element wise
                        if (rts.Shape[0] != arg.Shape[0] || rts.Shape[1] != arg.Shape[1]) {
                            if (rts.Shape[0] == arg.Shape[0] && arg.Shape[1] == 1) {
                                arg = arg.Copy();
                                arg = arg.ColExtend(rts.Shape[1]);
                                arg.Shape = [arg.Shape[0], rts.Shape[1]];
                            } else {
                                console.error("Element wise", opt);
                                return rts;
                            }
                        }
                        for (let i = 0; i < rts.Shape[0]; i++) {
                            for (let j = 0; j < rts.Shape[1]; j++) {
                                switch (opt) {
                                    case 0:
                                        rts.Values[i][j] = rts.Values[i][j] + arg.Values[i][j];
                                        break;
                                    case 1:
                                        rts.Values[i][j] = rts.Values[i][j] - arg.Values[i][j];
                                        break;
                                    case 2:
                                        rts.Values[i][j] = rts.Values[i][j] * arg.Values[i][j];
                                        break;
                                    case 3:
                                        rts.Values[i][j] = rts.Values[i][j] / arg.Values[i][j];
                                        break;
                                }
                            }
                        }
                        break;
                }
                break;
        }
        return rts;
    }

    // Model object
    class ModelObject {
        constructor() {
            this.Layers = [];
            this.Loss = {};
            this.Optimizer = {};
        }
        // Add layer
        AddLayer(config) {
            // Check config
            if (config == undefined) {
                return;
            }
            let inputSize = 1;
            let layer = new LayerObject();
            // Set default
            if (this.Layers.length == 0) { // First layer
                if (config.InputShape == undefined || config.InputShape.length == 0) {
                    return;
                }
                for (let i = 0; i < config.InputShape.length; i++) {
                    inputSize *= config.InputShape[i];
                }
                if (config.Units == undefined || config.Units == 0) {
                    config.Units = inputSize;
                }
            } else {
                // Not first layer
                if (config.Units == undefined || config.Units == 0) {
                    config.Units = this.Layers[this.Layers.length - 1].Units;
                }
                inputSize = this.Layers[this.Layers.length - 1].Units;
            }
            // Set activation function
            switch (config.Activation) {
                case "sigmoid":
                    layer.ActivationFunc = Sigmoid;
                    layer.DactivationFunc = Dsigmoid;
                    break;
                case "relu":
                    layer.ActivationFunc = Relu;
                    layer.DactivationFunc = Drelu;
                    break;
                default:
                    layer.ActivationFunc = Sigmoid;
                    layer.DactivationFunc = Dsigmoid;
            }
            layer.Units = config.Units;
            layer.InputSize = inputSize;
            // Add layer
            this.Layers.push(layer);
        }
        // Make model
        Make(config) {
            if (config == undefined) {
                return;
            }
            // Set loss function
            switch (config.Loss) {
                case "meanSquareError":
                    this.Loss = meanSquareError;
                    break;
                default:
                    this.Loss = meanSquareError;
            }
            // Set optimizer
            switch (config.Optimizer) {
                case "sgd":
                    this.Optimizer = gradientDescent;
                    break;
                default:
                    this.Optimizer = gradientDescent;
            }
            // Set learning rate
            if (config.LearningRate == undefined || config.LearningRate == 0 || typeof config.LearningRate != "number") {
                // Set default
                this.LearningRate = Tensor([], 0.2);
            } else {
                this.LearningRate = Tensor([], config.LearningRate);
            }
            // Create weights and biases in layers
            for (let i = 0; i < this.Layers.length; i++) {
                this.Layers[i].Weights = Tensor([this.Layers[i].Units, this.Layers[i].InputSize]).Random();
                this.Layers[i].Biases = Tensor([this.Layers[i].Units, 1]).Random();
            }
        }
        // Predict
        Predict(input) {
            if (input == undefined || input instanceof TensorObject == false) {
                // Not a tensor
                return;
            }
            let i = 0;
            // Forward propagation
            for (i = 0; i < this.Layers.length; i++) {
                this.Layers[i].WeightedSum = this.Layers[i].Weights.Mul(input).Add(this.Layers[i].Biases);
                this.Layers[i].Output = this.Layers[i].WeightedSum.Map(this.Layers[i].ActivationFunc);
                input = this.Layers[i].Output;
            }
            return this.Layers[i - 1].Output.Copy(); // Last layer
        }
        // Train
        Train(inputs, outputs, config) {
            // Check arguments
            if (inputs == undefined || inputs instanceof TensorObject == false || outputs == undefined || outputs instanceof TensorObject == false) {
                return;
            }
            // Check config
            if (config == undefined) {
                config = { BatchSize: 1, Epochs: 100 };
            }
            // Batch size
            if (config.BatchSize == undefined || typeof config.BatchSize != "number" || config.BatchSize < 1) {
                config.BatchSize = 1;
            }
            // Epochs
            if (config.Epochs == undefined || typeof config.Epochs != "number" || config.Epochs < 1) {
                config.Epochs = 100;
            }
            inputs = inputs.Transpose();
            outputs = outputs.Transpose();
            // Make batches
            let inputBatches = inputs.MakeBatches(config.BatchSize);
            let outputBatches = outputs.MakeBatches(config.BatchSize);
            // For each epochs
            for (let i = 0; i < config.Epochs; i++) {
                // If shuffle
                if (config.Shuffle == true) {
                    shuffle(inputBatches, outputBatches);
                }
                // For each batch
                for (let b = 0; b < inputBatches.length; b++) {
                    // Take a batch and predict
                    let batchOutput = this.Predict(inputBatches[b]);
                    let batchSize = Tensor([], inputBatches[b].Shape[1]);
                    if (config.EachEpoch != undefined && typeof config.EachEpoch == "function") {
                        config.EachEpoch(this.Loss(batchOutput, outputBatches[b], batchSize).AddAll(), i, b);
                    }
                    // Run optimizer
                    this.Optimizer(this, batchOutput, batchSize, inputBatches[b], outputBatches[b]);
                }
            }
        }
    }

    // Layer object
    class LayerObject {
        constructor() {
            this.Weights = [];
            this.Dweights = [];
            this.Biases = [];
            this.Dbiases = [];
            this.Output = [];
            this.WeightedSum = [];
            this.Units = 1;
            this.ActivationFunc = {};
            this.DactivationFunc = {};
            this.InputSize = 1;
            this.Err = [];
        }
    }

    // Shuffle
    function shuffle(arg1, arg2) {
        for (let i = 0; i < arg1.length; i++) {
            let p1 = Random(0, arg1.length, true);
            let p2 = Random(0, arg1.length, true);
            let temp1 = arg1[p1];
            let temp2 = arg2[p1];
            arg1[p1] = arg1[p2];
            arg2[p1] = arg2[p2];
            arg1[p2] = temp1;
            arg2[p2] = temp2;
        }
    }

    // Gradient descent
    function gradientDescent(m, batchOutput, batchSize, inputArg, output) {
        let lIndex = m.Layers.length - 1;
        m.Layers[lIndex].Err = batchOutput.Sub(output);
        // error . d(z)
        let errD = m.Layers[lIndex].Err.Mul(m.Layers[lIndex].Output.Map(m.Layers[lIndex].DactivationFunc)).Div(batchSize);
        m.Layers[lIndex].Dbiases = errD.AddCols();
        let input = inputArg;
        // error . d(z) * T(I)
        if (m.Layers.length > 1) {
            input = m.Layers[lIndex - 1].Output;
        }
        m.Layers[lIndex].Dweights = errD.Mul(input.Transpose());
        m.Layers[lIndex].Weights = m.Layers[lIndex].Weights.Sub(m.Layers[lIndex].Dweights.Mul(m.LearningRate));
        m.Layers[lIndex].Biases = m.Layers[lIndex].Biases.Sub(m.Layers[lIndex].Dbiases.Mul(m.LearningRate));
        //Back propagation
        for (let j = lIndex - 1; j >= 0; j--) {
            m.Layers[j].Err = m.Layers[j + 1].Weights.Transpose().Mul(m.Layers[j + 1].Err).Mul(m.Layers[j].Output.Map(m.Layers[j].DactivationFunc)).Div(batchSize);
            // error * T(I)
            if (j == 0) {
                input = inputArg;
            } else {
                input = m.Layers[j - 1].Output;
            }
            m.Layers[j].Dbiases = m.Layers[j].Err.AddCols();
            m.Layers[j].Dweights = m.Layers[j].Err.Mul(input.Transpose());
            m.Layers[j].Weights = m.Layers[j].Weights.Sub(m.Layers[j].Dweights.Mul(m.LearningRate));
            m.Layers[j].Biases = m.Layers[j].Biases.Sub(m.Layers[j].Dbiases.Mul(m.LearningRate));

        }
    }

    // Exported model function
    function Model() {
        return new ModelObject();
    }

    // Export 
    $.lsMl = {
        Tensor: Tensor,
        Random: Random,
        Model: Model,
        Sigmoid: Sigmoid,
        Dsigmoid: Dsigmoid,
        Relu: Relu,
        Drelu: Drelu,
        Shuffle: shuffle
    }
}(window));