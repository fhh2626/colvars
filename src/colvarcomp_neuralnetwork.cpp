// -*- c++ -*-

// This file is part of the Collective Variables module (Colvars).
// The original version of Colvars and its updates are located at:
// https://github.com/Colvars/colvars
// Please update all Colvars source files before making any changes.
// If you wish to distribute your changes, please submit them to the
// Colvars repository at GitHub.
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>
#include <fstream>
#include "colvarmodule.h"
#include "colvarvalue.h"
#include "colvar.h"
#include "colvarcomp.h"
#include "colvar_neuralnetworkcompute.h"

using namespace neuralnetworkCV;
bool startswith(std::string long_str, std::string short_str)
{
	int len=short_str.length();
	for(int i=0;i<len;i++)
	{
		if(long_str[i]!=short_str[i])
		{
			return false;
		}
	}
	
	return true;
}

colvar::neuralNetwork::neuralNetwork()
{
    set_function_type("neuralNetwork");
}


int colvar::neuralNetwork::init(std::string const &conf)
{
    int error_code = linearCombination::init(conf);
    if (error_code != COLVARS_OK) return error_code;
    // the output of neural network consists of multiple values
    // read "output_component" key to determine it
    get_keyval(conf, "output_component", m_output_index);
    // read type of each layer
    //record the type of each layer
    std::vector<LayerType> layer_types;
    std::string layer_type_file;
    get_keyval(conf,"Layer_Types",layer_type_file, std::string(""));
    //std::cout<<layer_type_file<<std::endl;
    std::string line;
    std::ifstream ifs_types(layer_type_file.c_str());
    while (std::getline(ifs_types, line)) {
        //std::cout<<line<<std::endl;
        if(startswith(line, "dense"))
        {
            layer_types.push_back(LayerType::DENSE);
        }
        else if(startswith(line, "colvar_attention"))
        {
            layer_types.push_back(LayerType::SELFATTENTION);
        }else
        {
            cvm::error("Invalid layer. This may not be accomplished. Please name your layer accordingly. ");
        }
    }
    // read weight files
    bool has_weight_files = true;
    size_t num_layers_weight = 0;
    std::vector<std::string> weight_files;
    while (has_weight_files) {
        std::string lookup_key = std::string{"layer"} + cvm::to_str(num_layers_weight + 1) + std::string{"_WeightsFile"};
        if (key_lookup(conf, lookup_key.c_str())) {
            std::string weight_filename;
            get_keyval(conf, lookup_key.c_str(), weight_filename, std::string(""));
            weight_files.push_back(weight_filename);
            cvm::log(std::string{"Will read layer["} + cvm::to_str(num_layers_weight + 1) + std::string{"] weights from "} + weight_filename + '\n');
            ++num_layers_weight;
        } else {
            has_weight_files = false;
        }
    }
    // read bias files
    bool has_bias_files = true;
    size_t num_layers_bias = 0;
    std::vector<std::string> bias_files;
    while (has_bias_files) {
        std::string lookup_key = std::string{"layer"} + cvm::to_str(num_layers_bias + 1) + std::string{"_BiasesFile"};
        if (key_lookup(conf, lookup_key.c_str())) {
            std::string bias_filename;
            get_keyval(conf, lookup_key.c_str(), bias_filename, std::string(""));
            bias_files.push_back(bias_filename);
            cvm::log(std::string{"Will read layer["} + cvm::to_str(num_layers_bias + 1) + std::string{"] biases from "} + bias_filename + '\n');
            ++num_layers_bias;
        } else {
            has_bias_files = false;
        }
    }
    // read activation function strings
    bool has_activation_functions = true;
    size_t num_activation_functions = 0;
    // pair(is_custom_function, function_string)
    std::vector<std::pair<bool, std::string>> activation_functions;
    while (has_activation_functions) {
        std::string lookup_key = std::string{"layer"} + cvm::to_str(num_activation_functions + 1) + std::string{"_activation"};
        std::string lookup_key_custom = std::string{"layer"} + cvm::to_str(num_activation_functions + 1) + std::string{"_custom_activation"};
        if (key_lookup(conf, lookup_key.c_str())) {
            // Ok, this is not a custom function
            std::string function_name;
            get_keyval(conf, lookup_key.c_str(), function_name, std::string(""));
            if (activation_function_map.find(function_name) == activation_function_map.end()) {
                return cvm::error("Unknown activation function name: \"" + function_name + "\".\n",
                               COLVARS_INPUT_ERROR);
            }
            activation_functions.push_back(std::make_pair(false, function_name));
            cvm::log(std::string{"The activation function for layer["} + cvm::to_str(num_activation_functions + 1) + std::string{"] is "} + function_name + '\n');
            ++num_activation_functions;
#ifdef LEPTON
        } else if (key_lookup(conf, lookup_key_custom.c_str())) {
            std::string function_expression;
            get_keyval(conf, lookup_key_custom.c_str(), function_expression, std::string(""));
            activation_functions.push_back(std::make_pair(true, function_expression));
            cvm::log(std::string{"The custom activation function for layer["} + cvm::to_str(num_activation_functions + 1) + std::string{"] is "} + function_expression + '\n');
            ++num_activation_functions;
#endif
        } else {
            has_activation_functions = false;
        }
    }
    //attention layer params
    bool has_attention_hyperparams=true;
    size_t num_attention_hyperparam=0;
    bool has_attention_weights=true;
    size_t num_attention_weights=0;
    bool has_attention_biases=true;
    size_t num_attention_biases=0;
    std::vector<std::string> hyperparam_files;
    std::vector<std::string> attention_weights_files;
    std::vector<std::string> attention_biases_files;
    //read weights of atttention layers
    while (has_attention_weights) {
        std::string lookup_key = std::string{"Attention"} + cvm::to_str(num_attention_weights + 1) + std::string{"_WeightsFile"};
        if(key_lookup(conf,lookup_key.c_str())){
            std::string weights_filename;
            get_keyval(conf, lookup_key.c_str(), weights_filename, std::string(""));
            attention_weights_files.push_back(weights_filename);
            cvm::log(std::string{"Will read self-attention layer["} + cvm::to_str(num_attention_weights + 1) + std::string{"] weights from "} + weights_filename + '\n');
            num_attention_weights++;
        }else{
                has_attention_weights=false;
        }
    }
    //read biases of atttention layers
    while (has_attention_biases) {
        std::string lookup_key = std::string{"Attention"} + cvm::to_str(num_attention_biases + 1) + std::string{"_BiasesFile"};
        if(key_lookup(conf,lookup_key.c_str())){
            std::string bias_filename;
            get_keyval(conf, lookup_key.c_str(), bias_filename, std::string(""));
            attention_biases_files.push_back(bias_filename);
            cvm::log(std::string{"Will read self-attention layer["} + cvm::to_str(num_attention_biases + 1) + std::string{"] biases from "} + bias_filename + '\n');
            num_attention_biases++;
        }else{
                has_attention_biases=false;
        }
    }
    //std::cout<<"read biase files of attention layer "<<std::endl;
    //read biases of atttention layers
    while (has_attention_hyperparams) {
        std::string lookup_key = std::string{"Attention"} + cvm::to_str(num_attention_hyperparam + 1) + std::string{"_HyperparamFile"};
        if(key_lookup(conf,lookup_key.c_str())){
            std::string hyperparam_filename;
            get_keyval(conf, lookup_key.c_str(), hyperparam_filename, std::string(""));
            hyperparam_files.push_back(hyperparam_filename);
            cvm::log(std::string{"Will read self-attention layer["} + cvm::to_str(num_attention_hyperparam + 1) + std::string{"] hyperparameters from "} + hyperparam_filename + '\n');
            num_attention_hyperparam++;
        }else{
                has_attention_hyperparams=false;
        }
    }
    // expect the three numbers are equal
    if ((num_layers_weight != num_layers_bias) || (num_layers_bias != num_activation_functions)) {
        return cvm::error(
            "Error: the numbers of weights, biases and activation functions do not match.\n",
            COLVARS_INPUT_ERROR);
    }
    if ((num_attention_weights != num_attention_biases) || (num_attention_biases != num_attention_hyperparam)) {
        cvm::error("Error: the number of weights, biases and hyperparameters for self-attention layer do not match.\n");
    }
    num_attention_weights=0;
    int i_layer=0;
//     nn = std::make_unique<neuralnetworkCV::neuralNetworkCompute>();
    // std::make_unique is only available in C++14
    if (nn) nn.reset();
    nn = std::unique_ptr<neuralnetworkCV::neuralNetworkCompute>(new neuralnetworkCV::neuralNetworkCompute());
    for(LayerType iter:layer_types)
    {
        if(iter==LayerType::DENSE)
        {
        denseLayer d;
#ifdef LEPTON
        if (activation_functions[i_layer].first) {
            // use custom function as activation function
            try {
                d = denseLayer(weight_files[i_layer], bias_files[i_layer], activation_functions[i_layer].second);
            } catch (std::exception &ex) {
                return cvm::error("Error on initializing layer " + cvm::to_str(i_layer) +
                                           " (" + ex.what() + ")\n",
                                       COLVARS_INPUT_ERROR);
            }
        } else {
#endif
            // query the map of supported activation functions
            const auto& f = activation_function_map[activation_functions[i_layer].second].first;
            const auto& df = activation_function_map[activation_functions[i_layer].second].second;
            try {
                d = denseLayer(weight_files[i_layer], bias_files[i_layer], f, df);
            } catch (std::exception &ex) {
                return cvm::error("Error on initializing layer " + cvm::to_str(i_layer) +
                                           " (" + ex.what() + ")\n",
                                       COLVARS_INPUT_ERROR);
            }
#ifdef LEPTON
        }
#endif
        // add a new dense layer to network
        if (nn->addDenseLayer(d)) {
            if (cvm::debug()) {
                // show information about the neural network
                cvm::log("Layer " + cvm::to_str(i_layer) + " : has " + cvm::to_str(d.getInputSize()) + " input nodes and " + cvm::to_str(d.getOutputSize()) + " output nodes.\n");
                for (size_t i_output = 0; i_output < d.getOutputSize(); ++i_output) {
                    for (size_t j_input = 0; j_input < d.getInputSize(); ++j_input) {
                        cvm::log("    weights[" + cvm::to_str(i_output) + "][" + cvm::to_str(j_input) + "] = " + cvm::to_str(d.getWeight(i_output, j_input)));
                    }
                    cvm::log("    biases[" + cvm::to_str(i_output) + "] = " + cvm::to_str(d.getBias(i_output)) + "\n");
                }
            }
        } else {
            return cvm::error("Error: error on adding a new dense layer.\n", COLVARS_INPUT_ERROR);
        }
        i_layer++;
    }else if(iter==LayerType::SELFATTENTION)
        {
            AttentionLayer d = AttentionLayer(attention_weights_files[num_attention_weights], attention_biases_files[num_attention_weights], hyperparam_files[num_attention_weights]);
            
            // add a new attention layer to network
            if (nn->addAttentionLayer(d)) {
                // show information about the neural network
                cvm::log("Attention layer " + cvm::to_str(num_attention_weights)+"\n");
                std::string outputbuf="";
                cvm::log("Matrix WQ with bias: \n");
                for(size_t i=0;i<d.getD_model();i++)
                {
                    
                    for(size_t j=0;j<d.getD_model();j++)
                    {
                        outputbuf.append(std::to_string(d.getWeight_q(i,j))+"\t");
                    }
                    outputbuf.append("\n");
                    cvm::log(outputbuf);
                    outputbuf="";
                    outputbuf.append(std::to_string(d.getBias_q(i))+"\t");
                }
                cvm::log(outputbuf);
                outputbuf="";
                cvm::log("Matrix WQ: \n");
                for(size_t i=0;i<d.getD_model();i++)
                {
                    
                    for(size_t j=0;j<d.getD_model();j++)
                    {
                        outputbuf.append(std::to_string(d.getWeight_k(i,j))+"\t");
                    }
                    outputbuf.append("\n");
                    cvm::log(outputbuf);
                    outputbuf="";
                    outputbuf.append(std::to_string(d.getBias_k(i))+"\t");
                }
                cvm::log(outputbuf);
                outputbuf="";
                cvm::log("Matrix WQ: \n");
                for(size_t i=0;i<d.getD_model();i++)
                {
                    
                    for(size_t j=0;j<d.getD_model();j++)
                    {
                        outputbuf.append(std::to_string(d.getWeight_v(i,j))+"\t");
                    }
                    outputbuf.append("\n");
                    cvm::log(outputbuf);
                    outputbuf="";
                    outputbuf.append(std::to_string(d.getBias_v(i))+"\t");
                }
                cvm::log(outputbuf);
                outputbuf="";
                cvm::log("Matrix WQ: \n");
                for(size_t i=0;i<d.getD_model();i++)
                {
                    
                    for(size_t j=0;j<d.getD_model();j++)
                    {
                        outputbuf.append(std::to_string(d.getWeight_o(i,j))+"\t");
                    }
                    outputbuf.append("\n");
                    cvm::log(outputbuf);
                    outputbuf="";
                    outputbuf.append(std::to_string(d.getBias_o(i))+"\t");
                }
                cvm::log(outputbuf);
                outputbuf="";
            } else {
            cvm::error("Error: error on adding a new self attention layer.\n");
            }
            num_attention_weights++;
        } else
        {
            cvm::error("Invalid layer. This may not been accomplished.");
        }
    }
    nn->input().resize(cv.size());
    return error_code;
}

colvar::neuralNetwork::~neuralNetwork() {
}

void colvar::neuralNetwork::calc_value() {
    x.reset();
    for (size_t i_cv = 0; i_cv < cv.size(); ++i_cv) {
        cv[i_cv]->calc_value();
        const colvarvalue& current_cv_value = cv[i_cv]->value();
        // for current nn implementation we have to assume taht types are always scaler
        if (current_cv_value.type() == colvarvalue::type_scalar) {
            nn->input()[i_cv] = cv[i_cv]->sup_coeff * (cvm::pow(current_cv_value.real_value, cv[i_cv]->sup_np));
        } else {
            cvm::error("Error: using of non-scaler component.\n");
            return;
        }
    }
    nn->compute();
    x = nn->getOutput(m_output_index);
}

void colvar::neuralNetwork::calc_gradients() {
    for (size_t i_cv = 0; i_cv < cv.size(); ++i_cv) {
        cv[i_cv]->calc_gradients();
        if (cv[i_cv]->is_enabled(f_cvc_explicit_gradient)) {
            const cvm::real factor = nn->getGradient(m_output_index, i_cv);
            const cvm::real factor_polynomial = getPolynomialFactorOfCVGradient(i_cv);
            for (size_t j_elem = 0; j_elem < cv[i_cv]->value().size(); ++j_elem) {
                for (size_t k_ag = 0 ; k_ag < cv[i_cv]->atom_groups.size(); ++k_ag) {
                    for (size_t l_atom = 0; l_atom < (cv[i_cv]->atom_groups)[k_ag]->size(); ++l_atom) {
                        (*(cv[i_cv]->atom_groups)[k_ag])[l_atom].grad = factor_polynomial * factor * (*(cv[i_cv]->atom_groups)[k_ag])[l_atom].grad;
                    }
                }
            }
        }
    }
}

void colvar::neuralNetwork::apply_force(colvarvalue const &force) {
    for (size_t i_cv = 0; i_cv < cv.size(); ++i_cv) {
        // If this CV us explicit gradients, then atomic gradients is already calculated
        // We can apply the force to atom groups directly
        if (cv[i_cv]->is_enabled(f_cvc_explicit_gradient)) {
            for (size_t k_ag = 0 ; k_ag < cv[i_cv]->atom_groups.size(); ++k_ag) {
                (cv[i_cv]->atom_groups)[k_ag]->apply_colvar_force(force.real_value);
            }
        } else {
            // Compute factors for polynomial combinations
            const cvm::real factor_polynomial = getPolynomialFactorOfCVGradient(i_cv);
            const cvm::real factor = nn->getGradient(m_output_index, i_cv);;
            colvarvalue cv_force = force.real_value * factor * factor_polynomial;
            cv[i_cv]->apply_force(cv_force);
        }
    }
}


cvm::real colvar::neuralNetwork::dist2(colvarvalue const &x1, colvarvalue const &x2) const
{
  return x1.dist2(x2);
}


colvarvalue colvar::neuralNetwork::dist2_lgrad(colvarvalue const &x1, colvarvalue const &x2) const
{
  return x1.dist2_grad(x2);
}


colvarvalue colvar::neuralNetwork::dist2_rgrad(colvarvalue const &x1, colvarvalue const &x2) const
{
  return x2.dist2_grad(x1);
}



void colvar::neuralNetwork::wrap(colvarvalue & /* x_unwrapped */) const {}
