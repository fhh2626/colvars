// -*- Mode:c++; c-basic-offset: 4; -*-

// This file is part of the Collective Variables module (Colvars).
// The original version of Colvars and its updates are located at:
// https://github.com/Colvars/colvars
// Please update all Colvars source files before making any changes.
// If you wish to distribute your changes, please submit them to the
// Colvars repository at GitHub.

#include <iostream>
#include <fstream>
#include "colvar_neuralnetworkcompute.h"
#include "colvarparse.h"
#include "colvarproxy.h"

namespace neuralnetworkCV {
std::map<std::string, std::pair<std::function<double(double)>, std::function<double(double)>>> activation_function_map
{
    {"tanh",     {[](double x){return std::tanh(x);},
                  [](double x){return 1.0 - std::tanh(x) * std::tanh(x);}}},
    {"sigmoid",  {[](double x){return 1.0 / (1.0 + std::exp(-x));},
                  [](double x){return std::exp(-x) / ((1.0 + std::exp(-x)) * (1.0 + std::exp(-x)));}}},
    {"linear",   {[](double x){return x;},
                  [](double /*x*/){return 1.0;}}},
    {"relu",     {[](double x){return x < 0. ? 0. : x;},
                  [](double x){return x < 0. ? 0. : 1.;}}},
    {"lrelu100", {[](double x){return x < 0. ? 0.01 * x : x;},
                  [](double x){return x < 0. ? 0.01     : 1.;}}},
    {"elu",      {[](double x){return x < 0. ? std::exp(x)-1. : x;},
                  [](double x){return x < 0. ? std::exp(x)    : 1.;}}}
};

#ifdef LEPTON
customActivationFunction::customActivationFunction():
expression(), value_evaluator(nullptr), gradient_evaluator(nullptr),
input_reference(nullptr), derivative_reference(nullptr) {}

customActivationFunction::customActivationFunction(const std::string& expression_string):
expression(), value_evaluator(nullptr), gradient_evaluator(nullptr),
input_reference(nullptr), derivative_reference(nullptr) {
    setExpression(expression_string);
}

customActivationFunction::customActivationFunction(const customActivationFunction& source):
expression(), value_evaluator(nullptr), gradient_evaluator(nullptr),
input_reference(nullptr), derivative_reference(nullptr) {
    // check if the source object is initialized
    if (source.value_evaluator != nullptr) {
        this->setExpression(source.expression);
    }
}

customActivationFunction& customActivationFunction::operator=(const customActivationFunction& source) {
    if (source.value_evaluator != nullptr) {
        this->setExpression(source.expression);
    } else {
        expression = std::string();
        value_evaluator = nullptr;
        gradient_evaluator = nullptr;
        input_reference = nullptr;
        derivative_reference = nullptr;
    }
    return *this;
}

void customActivationFunction::setExpression(const std::string& expression_string) {
    expression = expression_string;
    Lepton::ParsedExpression parsed_expression;
    // the variable must be "x" for the input of an activation function
    const std::string activation_input_variable{"x"};
    // parse the expression
    try {
        parsed_expression = Lepton::Parser::parse(expression);
    } catch (...) {
        cvm::error("Error parsing or compiling expression \"" + expression + "\".\n", COLVARS_INPUT_ERROR);
    }
    // compile the expression
    try {
        value_evaluator = std::unique_ptr<Lepton::CompiledExpression>(new Lepton::CompiledExpression(parsed_expression.createCompiledExpression()));
    } catch (...) {
        cvm::error("Error compiling expression \"" + expression + "\".\n", COLVARS_INPUT_ERROR);
    }
    // create a compiled expression for the derivative
    try {
        gradient_evaluator = std::unique_ptr<Lepton::CompiledExpression>(new Lepton::CompiledExpression(parsed_expression.differentiate(activation_input_variable).createCompiledExpression()));
    } catch (...) {
        cvm::error("Error creating compiled expression for variable \"" + activation_input_variable + "\".\n", COLVARS_INPUT_ERROR);
    }
    // get the reference to the input variable in the compiled expression
    try {
        input_reference = &(value_evaluator->getVariableReference(activation_input_variable));
    } catch (...) {
        cvm::error("Error on getting the reference to variable \"" + activation_input_variable + "\" in the compiled expression.\n", COLVARS_INPUT_ERROR);
    }
    // get the reference to the input variable in the compiled derivative expression
    try {
        derivative_reference = &(gradient_evaluator->getVariableReference(activation_input_variable));
    } catch (...) {
        cvm::error("Error on getting the reference to variable \"" + activation_input_variable + "\" in the compiled derivative exprssion.\n", COLVARS_INPUT_ERROR);
    }
}

std::string customActivationFunction::getExpression() const {
    return expression;
}

double customActivationFunction::evaluate(double x) const {
    *input_reference = x;
    return value_evaluator->evaluate();
}

double customActivationFunction::derivative(double x) const {
    *derivative_reference = x;
    return gradient_evaluator->evaluate();
}
#endif

denseLayer::denseLayer(const std::string& weights_file, const std::string& biases_file, const std::function<double(double)>& f, const std::function<double(double)>& df): m_activation_function(f), m_activation_function_derivative(df) {
#ifdef LEPTON
    m_use_custom_activation = false;
#endif
    readFromFile(weights_file, biases_file);
}

#ifdef LEPTON
denseLayer::denseLayer(const std::string& weights_file, const std::string& biases_file, const std::string& custom_activation_expression) {
    m_use_custom_activation = true;
    m_custom_activation_function = customActivationFunction(custom_activation_expression);
    readFromFile(weights_file, biases_file);
}
#endif

void denseLayer::readFromFile(const std::string& weights_file, const std::string& biases_file) {
    // parse weights file
    m_weights.clear();
    m_biases.clear();
    std::string line;
    colvarproxy *proxy = cvm::main()->proxy;
    auto &ifs_weights = proxy->input_stream(weights_file, "weights file");
    while (std::getline(ifs_weights, line)) {
        if (!ifs_weights) {
            throw std::runtime_error("I/O error while reading " + weights_file);
        }
        std::vector<std::string> splitted_data;
        colvarparse::split_string(line, std::string{" "}, splitted_data);
        if (splitted_data.size() > 0) {
            std::vector<double> weights_tmp(splitted_data.size());
            for (size_t i = 0; i < splitted_data.size(); ++i) {
                try {
                    weights_tmp[i] = std::stod(splitted_data[i]);
                } catch (...) {
                    throw std::runtime_error("Cannot convert " + splitted_data[i] + " to a number while reading file " + weights_file);
                }
            }
            m_weights.push_back(weights_tmp);
        }
    }
    proxy->close_input_stream(weights_file);

    // parse biases file
    auto &ifs_biases = proxy->input_stream(biases_file, "biases file");
    while (std::getline(ifs_biases, line)) {
        if (!ifs_biases) {
            throw std::runtime_error("I/O error while reading " + biases_file);
        }
        std::vector<std::string> splitted_data;
        colvarparse::split_string(line, std::string{" "}, splitted_data);
        if (splitted_data.size() > 0) {
            double bias = 0;
            try {
                bias = std::stod(splitted_data[0]);
            } catch (...) {
                throw std::runtime_error("Cannot convert " + splitted_data[0] + " to a number while reading file " + biases_file);
            }
            m_biases.push_back(bias);
        }
    }
    proxy->close_input_stream(biases_file);

    m_input_size = m_weights[0].size();
    m_output_size = m_weights.size();
}

void denseLayer::setActivationFunction(const std::function<double(double)>& f, const std::function<double(double)>& df) {
    m_activation_function = f;
    m_activation_function_derivative = df;
}

void denseLayer::compute(const std::vector<double>& input, std::vector<double>& output) const {
    for (size_t i = 0; i < m_output_size; ++i) {
        output[i] = 0;
        for (size_t j = 0; j < m_input_size; ++j) {
            output[i] += input[j] * m_weights[i][j];
        }
        output[i] += m_biases[i];
#ifdef LEPTON
        if (m_use_custom_activation) {
            output[i] = m_custom_activation_function.evaluate(output[i]);
        } else {
#endif
            output[i] = m_activation_function(output[i]);
#ifdef LEPTON
        }
#endif
    }
}

double denseLayer::computeGradientElement(const std::vector<double>& input, const size_t i, const size_t j) const {
    double sum_with_bias = 0;
    for (size_t j_in = 0; j_in < m_input_size; ++j_in) {
        sum_with_bias += input[j_in] * m_weights[i][j_in];
    }
    sum_with_bias += m_biases[i];
#ifdef LEPTON
    if (m_use_custom_activation) {
        const double grad_ij = m_custom_activation_function.derivative(sum_with_bias) * m_weights[i][j];
        return grad_ij;
    } else {
#endif
        const double grad_ij = m_activation_function_derivative(sum_with_bias) * m_weights[i][j];
        return grad_ij;
#ifdef LEPTON
    }
#endif
}

void denseLayer::computeGradient(const std::vector<double>& input, std::vector<std::vector<double>>& output_grad) const {
    for (size_t j = 0; j < m_input_size; ++j) {
        for (size_t i = 0; i < m_output_size; ++i) {
            output_grad[i][j] = computeGradientElement(input, i, j);
        }
    }
}

neuralNetworkCompute::neuralNetworkCompute(const std::vector<denseLayer>& dense_layers): m_dense_layers(dense_layers) {
    m_layers_output.resize(m_dense_layers.size());
    m_grads_tmp.resize(m_dense_layers.size());
    for (size_t i_layer = 0; i_layer < m_layers_output.size(); ++i_layer) {
        m_layers_output[i_layer].assign(m_dense_layers[i_layer].getOutputSize(), 0);
        m_grads_tmp[i_layer].assign(m_dense_layers[i_layer].getOutputSize(), std::vector<double>(m_dense_layers[i_layer].getInputSize(), 0));
    }
}

std::vector<std::vector<double>> neuralNetworkCompute::multiply_matrix(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    const size_t m = A.size();
    const size_t n = B.size();
    if (A[0].size() != n) {
        std::cerr << "Error on multiplying matrices!\n";
    }
    const size_t t = B[0].size();
    std::vector<std::vector<double>> C(m, std::vector<double>(t, 0.0));
    for (size_t i = 0; i < m; ++i) {
        for (size_t k = 0; k < n; ++k) {
            const auto tmp = A[i][k];
            auto& C_i = C[i];
            auto& B_k = B[k];
            for (size_t j = 0; j < t; ++j) {
                C_i[j] += tmp * B_k[j];
            }
        }
    }
    return C;
}

//necesseray matrix calculate functions
bool startswith(std::string long_str, std::string short_str)
{
    /*
     * @param long_str longer string
     * @param short_str shorter string
     */
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
//create matrix with h rows and i columns
std::vector<std::vector<double>> creatematrix(int h,int l, double initial=0)
{
	std::vector<std::vector<double>> v(h, std::vector<double>(l, initial));
	return v;
}
//matrix plus
std::vector<std::vector<double>> plus(const std::vector<std::vector<double>>&A,const std::vector<std::vector<double>>&B)
{
	int h=A.size();
	int l=A[0].size();
    if(A.size()!=B.size()||A[0].size()!=B[0].size())
    {
        std::cout<<("Matrix plus is not regular. ");
    }
	std::vector<std::vector<double>> C=creatematrix( h, l); 
	for(int i=0;i<h;i++)
	{
		for (int j = 0; j < l; j++)
		{
			C[i][j]=A[i][j]+B[i][j];  
			//if (abs(C[i][j])<epsilon)
			//{
			//	C[i][j]=0.0;
			//}
		}
	}
	return C;
}
//Of cause minus function can be created
std::vector<std::vector<double>> minus(const std::vector<std::vector<double>>&A,const std::vector<std::vector<double>>&B)
{
	int h=A.size();
	int l=A[0].size();
    if(A.size()!=B.size()||A[0].size()!=B[0].size())
    {
        std::cout<<("Matrix minus is not regular. ");
    }
	std::vector<std::vector<double>> C=creatematrix( h, l);
	for(int i=0;i<h;i++)
	{
		for (int j = 0; j < l; j++)
		{
			C[i][j]=A[i][j]-B[i][j];
			//if (abs(C[i][j])<epsilon)
			//{
			//	C[i][j]=0.0;
			//}
		}
	}
	return C;
}
std::vector<std::vector<std::vector<double>>> plus(const std::vector<std::vector<std::vector<double>>>&A,const std::vector<std::vector<std::vector<double>>>&B)
{
    if(A.size()!=B.size())
    {
        std::cout<<("The depth of A and B not equals. ");
    }
	int d=A.size();
    std::vector<std::vector<std::vector<double>>> AT;
    for(int depth_i=0;depth_i<d;depth_i++)
    {
        AT.push_back(plus(A[depth_i],B[depth_i]));
    }
	return AT;
}
std::vector<std::vector<std::vector<double>>> minus(const std::vector<std::vector<std::vector<double>>>&A,const std::vector<std::vector<std::vector<double>>>&B)
{
    if(A.size()!=B.size())
    {
        std::cout<<("The depth of A and B not equals. ");
    }
	int d=A.size();
    std::vector<std::vector<std::vector<double>>> AT;
    for(int depth_i=0;depth_i<d;depth_i++)
    {
        AT.push_back(minus(A[depth_i],B[depth_i]));
    }
	return AT;
}
//matrix dot multiply
std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>&A,const std::vector<std::vector<double>>&B)
{
	int A_h=A.size();
	int A_l=A[0].size();
	int B_h=B.size();
	int B_l=B[0].size();
	if(A_l !=B_h)
	{
		std::cout<<("ERROR: Matrix multiple error. matmul("+std::to_string(A_h)+"*"+std::to_string(A_l)+","+std::to_string(B_h)+"*"+std::to_string(B_l)+") is not compatible. \n");
	}
	std::vector<std::vector<double>> C=creatematrix(A_h,B_l);
	for (int i = 0; i < A_h; i++)
	{
		for (int j = 0; j < B_l; j++)
		{
			C[i][j]=0;
			for (int k = 0; k < A_l; k++)
			{
				C[i][j] +=A[i][k]*B[k][j];
			}
		}
	}
	return C;
}
std::vector<std::vector<std::vector<double>>> matmul(const std::vector<std::vector<std::vector<double>>>& A, const std::vector<std::vector<std::vector<double>>>& B)
{
    std::vector<std::vector<std::vector<double>>> AT;
    if(A.size()!=B.size())
    {
        std::cout<<("The depth of A and B is not equal. ");
    }
    int d=A.size();
    for(int i=0;i<d;i++)
    {
        AT.push_back(neuralNetworkCompute::multiply_matrix(A[i], B[i]));
    }
    return AT;
}
//matrix multiplied by number
std::vector<std::vector<double>> multiply_num(const std::vector<std::vector<double>>&A,double num)
{
	int A_h=A.size();
	int A_l=A[0].size();
	std::vector<std::vector<double>> B=creatematrix(A_h,A_l);
	for (int i = 0; i < A_h; i++)
	{
		for (int j = 0; j < A_l; j++)
		{
			B[i][j]=num*A[i][j];
		}
	}
	return B;
}
//3 dimension version
std::vector<std::vector<std::vector<double>>> multiply_num(const std::vector<std::vector<std::vector<double>>>&A,double num)
{
    int A_d=A.size();
	std::vector<std::vector<std::vector<double>>> B;
    for(int i=0;i<A_d;i++)
    {
        B.push_back(multiply_num(A[i],num));
    }
	return B;
}
//vertical stack the matrix
std::vector<std::vector<double>> vstack(const std::vector<std::vector<double>>&A,const std::vector<std::vector<double>>&B)
{
	int A_h=A.size();
	int A_l=A[0].size();
	int B_h=B.size();
	int B_l=B[0].size();
	if (A_l != B_l)
	{
		std::cout<<("ERROR: Matrix vertical stack error. vstack("+std::to_string(A_h)+"*"+std::to_string(A_l)+","+std::to_string(B_h)+"*"+std::to_string(B_l)+") is not compatible. \n");
	}
	auto C(A);
    C.insert(C.end(), B.begin(), B.end());
    return C;
}
//horizontal stack the matrix
std::vector<std::vector<double>> hstack(const std::vector<std::vector<double>>&A,const std::vector<std::vector<double>>&B)
{
	int A_h=A.size();
	int A_l=A[0].size();
	int B_h=B.size();
	int B_l=B[0].size();
	if (A_h != B_h)
	{
		std::cout<<("ERROR: Matrix horizontal stack error. hstack("+std::to_string(A_h)+"*"+std::to_string(A_l)+","+std::to_string(B_h)+"*"+std::to_string(B_l)+") is not compatible. \n");
	}
	auto C(A);
	for (int i = 0; i < B_h; i++)
	{
        C[i].insert(C[i].end(), B[i].begin(), B[i].end());
	}
	return C;
}
//transpose the matrix
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>> &A)
{
	std::vector<std::vector<double>> AT=creatematrix(A[0].size(),A.size());
	int h=AT.size();
	int l=AT[0].size();
	for (int i = 0; i <h ; i++)
	{
		for (int j = 0; j < l; j++)
		{
			AT[i][j]=A[j][i];
		}
	}
	return AT;
}
//transpose the third dimension of matrix with each omitting the first corderate
std::vector<std::vector<std::vector<double>>> transpose(const std::vector<std::vector<std::vector<double>>>& A)
{
	std::vector<std::vector<std::vector<double>>> AT;
	for(unsigned int i=0;i<A.size();i++)
	{
		AT.push_back(transpose(A[i]));
	}
	return AT;
}
void show_matrix(const std::vector<std::vector<double>> &A)
{
	int h=A.size();
	int l=A[0].size();
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < l; j++)
		{
			std::cout<<A[i][j]<<"\t";
		}
		std::cout<<std::endl;
	}
}
//reshape the matrix
//from one dimention to two dimention
std::vector<std::vector<double>> reshape(const std::vector<double>& A, int shape0,int shape1)
{
    std::vector<std::vector<double>> AT=creatematrix(shape0,shape1);
    if(A.size()/shape0!=(unsigned int) shape1)
    {
        std::cout<<("ERROR: Invalid reshape with from one dimention to"+std::to_string(shape0)+"*"+std::to_string(shape1)+"\n");
        return AT;
    }
    for(int i=0;i<shape0;i++)
    {
        for(int j=0;j<shape1;j++)
        {
            AT[i][j]=A[i*shape1+j];
        }
    }
    return AT;
}
//from two dimention to two dimention
std::vector<std::vector<double>> reshape(const std::vector<std::vector<double>>& A, int shape0,int shape1)
{
    std::vector<std::vector<double>> AT=creatematrix(shape0,shape1);
    if(A.size()*A[0].size()/shape0!=(unsigned int)shape1)
    {
        std::cout<<("ERROR: Invalid reshape with from two dimention to"+std::to_string(shape0)+"*"+std::to_string(shape1)+"\n");
        return AT;
    }
    for(int i=0;i<shape0;i++)
    {
        for(int j=0;j<shape1;j++)
        {
            AT[i][j]=A[(i+j)/A[0].size()][(i+j)%A[0].size()];
        }
    }
    return AT;
}
/**
 * Unfold a 2D vector into 1D vector in column dominated order
 * Normal function concat() in pytorch is to in row dominated order
 * Notice that this is a specific function only applied into this model, because it is reversive. 
 * that is, to concatenate each seq_len*d_model matrix. 
*/
std::vector<double> concat(const std::vector<std::vector<double>>& A)
{
    std::vector<double> AT(A.size()*A[0].size());
    int seq_len=A.size();
    int d_model=A[0].size();

    for(int i=0;i<d_model;i++)
    {
        for(int j=0;j<seq_len;j++)
        {
            AT[i*seq_len+j]=A[j][i];
        }
    }
    return AT;
}
//from two dimention to three dimention, which used as multihead
std::vector<std::vector<std::vector<double>>> reshape(const std::vector<std::vector<double>> A, int num_head, bool split_head)
{
	if(split_head==true)
	{
    int depth=A[0].size()/num_head;
    std::vector<std::vector<std::vector<double>>> AT;
    std::vector<std::vector<double>> buf=creatematrix(A.size(),depth);
    for(int iter=0;iter<num_head;iter++)
    {
        for(unsigned int i=0;i<A.size();i++)
        {
            for(int j=0;j<depth;j++)
            {
                buf[i][j]=A[i][j+iter*depth];
            }
        }
        AT.push_back(buf);
    }
    return AT;
	}
	else
	{
	int depth=A.size()/num_head;
	std::vector<std::vector<std::vector<double>>> AT;
	for(int iter=0;iter<num_head;iter++)
	{
		std::vector<std::vector<double>> buf;
		for(int i=0;i<depth;i++)
		{
			buf.push_back(A[i+iter*depth]);
		}
		AT.push_back(buf);
	}
	return AT;
	}
}
//concatenate the 3 dimention matrix into 2 dimention matrix. Not applicatable to other conditions. 
std::vector<std::vector<double>> reshape(const std::vector<std::vector<std::vector<double>>>& A)
{
    int depth=A.size();
    std::vector<std::vector<double>> AT=A[0];
    for(int i=1;i<depth;i++)
    {
        AT=hstack(AT,A[i]);
    }
    return AT;
}
//softmax where axit = -1
std::vector<double> softmax(const std::vector<double>& A)
{
    double total=0;
	double MAX=A[0];
	for(auto x:A)
	{
		MAX=std::max(x,MAX);
	}
    std::vector<double> result;
    for(auto x:A)
    {
        result.push_back(exp(x-MAX));
        total += result.back();
    }
    for(unsigned int i = 0; i < A.size(); i++)
    {
        result[i] /= total;
    }
	return result;
}

std::vector<std::vector<std::vector<double>>> softmax(const std::vector<std::vector<std::vector<double>>>& A, int axis=-1)
{
    //axis ignored because it haven't been implemented. 
    std::vector<std::vector<std::vector<double>>> AT;
    
    for(unsigned int i=0;i<A.size();i++)
    {
        std::vector<std::vector<double>> buf;
        for(unsigned int j=0;j<A[0].size();j++)
        {
            buf.push_back(softmax(A[i][j]));
        }
        AT.push_back(buf);
    }
    return AT;
}
//create diag(A) whose dimention is (A.size(), A.size())
std::vector<std::vector<double>> diag_matrix(const std::vector<double>& A)
{
    std::vector<std::vector<double>> AT=creatematrix(A.size(),A.size());
    for(unsigned int i=0;i<A.size();i++)
    {
        AT[i][i]=A[i];
    }
    return AT;
}
//input 1 dimention array and get matrix of n*n
std::vector<std::vector<double>> softmax_derivative(const std::vector<double>& A)
{
    std::vector<std::vector<double>> AT=creatematrix(A.size(),A.size());
    //if(A[0].size()!=1)
    //{
    //    std::cout<<("softmax_derivative error, where input is a matrix. ");
    //    return AT;
    //}
    std::vector<double> sof_A=softmax(A);
    std::vector<std::vector<double>> diag_sof_A=diag_matrix(sof_A);
    std::vector<std::vector<double>> mul_sof_A=matmul(reshape(sof_A,A.size(),1),transpose(reshape(sof_A,A.size(),1)));
    return minus(diag_sof_A,mul_sof_A);
}
//create one hot matrix with particular cordinate assigned as 1
std::vector<std::vector<double>> one_hot_matrix(const std::vector<std::vector<double>> X, int row, int column)
{
    std::vector<std::vector<double>> AT=creatematrix(X.size(),X[0].size());
    AT[row][column]=1;
    return AT;
} 
std::vector<double> arange(double start, double stop, double step)
{
    std::vector<double> result;
    while(start<stop)
    {
        result.push_back(start);
        start+=step;

    }
    return result;
}
//split a matrix from column start to column stop, not include column stop. 
std::vector<std::vector<double>> hsplit(std::vector<std::vector<double>>& A, int start, int stop)
{
    std::vector<std::vector<double>> AT=creatematrix(A.size(),stop-start);
    for(unsigned int i=0;i<A.size();i++)
    {
        for(int j=start;j<stop;j++)
        {
            AT[i][j-start]=A[i][j];
        }
    }
    return AT;
}

AttentionLayer::AttentionLayer(const std::string& weights_file, const std::string& biases_file, const std::string& hyperparams_file)
{
    readFromFile(weights_file, biases_file,hyperparams_file);
}

void AttentionLayer::readFromFile(const std::string& weights_file, const std::string& biases_file, const std::string& hyperparams_file) 
{
    // parse weights file
    m_q_weights.clear();
    m_q_biases.clear();
    m_k_weights.clear();
    m_k_biases.clear();
    m_v_weights.clear();
    m_v_biases.clear();
    m_o_weights.clear();
    m_o_biases.clear();
    std::string line;
    //read hyperparameters
    std::ifstream ifs_hyper(hyperparams_file.c_str());
    while(std::getline(ifs_hyper, line))
    {
        if (!ifs_hyper) {
            throw std::runtime_error("I/O error while reading " + hyperparams_file);
        }
        std::vector<std::string> splited_data;
        colvarparse::split_string(line,std::string(":"),splited_data);
        try{
            if(startswith(line,"num_colvars"))
            {
                m_input_size=std::stod(splited_data[1]);
            }else if(startswith(line,"d_model"))
            {
                m_d_model=std::stod(splited_data[1]);
            }else if(startswith(line,"num_head"))
            {
                m_num_head=std::stod(splited_data[1]);
            }else
            {
                cvm::error("No hyperparameter called "+splited_data[0] + ", omitted. ");
            }
        } catch (...) {
            throw std::runtime_error("Cannot convert " + splited_data[1] + " to a number while reading file " + hyperparams_file);
        }
    }
    //read wq, wk, wv, wo
    std::ifstream ifs_weights(weights_file.c_str());
    for(size_t i=0;i<getD_model();i++)
    {
        std::getline(ifs_weights,line);
        if(startswith(line,"#"))
        {
            i--;
            continue;
        }
        std::vector<std::string> splited_data;
        colvarparse::split_string(line, std::string{" "}, splited_data);
        if(getD_model()!=splited_data.size())
        {
            cvm::error("Split data error. Please check the input weight file with delimiter of \" \".");
        }
        std::vector<double> weights_tmp(splited_data.size());
        for (size_t i = 0; i < splited_data.size(); ++i) 
        {
            try{
                weights_tmp[i] = std::stod(splited_data[i]);
            } catch (...) {
            throw std::runtime_error("Cannot convert " + splited_data[i] + " to a number while reading file " + weights_file);
            }
        }
        m_q_weights.push_back(weights_tmp);
    }
    for(size_t i=0;i<getD_model();i++)
    {
        std::getline(ifs_weights,line);
        if(startswith(line,"#"))
        {
            i--;
            continue;
        }
        std::vector<std::string> splited_data;
        colvarparse::split_string(line, std::string{" "}, splited_data);
        
        std::vector<double> weights_tmp(splited_data.size());
        for (size_t i = 0; i < splited_data.size(); ++i) 
        {
            try{
                weights_tmp[i] = std::stod(splited_data[i]);
            } catch (...) {
            throw std::runtime_error("Cannot convert " + splited_data[i] + " to a number while reading file " + weights_file);
            }
        }
        m_k_weights.push_back(weights_tmp);
    }
    for(size_t i=0;i<getD_model();i++)
    {
        std::getline(ifs_weights,line);
        if(startswith(line,"#"))
        {
            i--;
            continue;
        }
        std::vector<std::string> splited_data;
        colvarparse::split_string(line, std::string{" "}, splited_data);
        
        std::vector<double> weights_tmp(splited_data.size());
        for (size_t i = 0; i < splited_data.size(); ++i) 
        {
            try{
                weights_tmp[i] = std::stod(splited_data[i]);
            } catch (...) {
            throw std::runtime_error("Cannot convert " + splited_data[i] + " to a number while reading file " + weights_file);
            }
        }
        m_v_weights.push_back(weights_tmp);
    }
    for(size_t i=0;i<getD_model();i++)
    {
        std::getline(ifs_weights,line);
        if(startswith(line,"#"))
        {
            i--;
            continue;
        }
        std::vector<std::string> splited_data;
        colvarparse::split_string(line, std::string{" "}, splited_data);
        
        std::vector<double> weights_tmp(splited_data.size());
        for (size_t i = 0; i < splited_data.size(); ++i) 
        {
            try{
                weights_tmp[i] = std::stod(splited_data[i]);
            } catch (...) {
            throw std::runtime_error("Cannot convert " + splited_data[i] + " to a number while reading file " + weights_file);
            }
        }
        m_o_weights.push_back(weights_tmp);
    }
    // parse biases file
    std::ifstream ifs_biases(biases_file.c_str());
    for(size_t i=0;i<getD_model();i++)
    {
        std::getline(ifs_biases, line);
        if(startswith(line,"#"))
        {
            i--;
            continue;
        }
        try{
            m_q_biases.push_back(std::stod(line));
        } catch (...) {
            throw std::runtime_error("Cannot convert " + line + " to a number while reading file " + biases_file);
        }
    }
    for(size_t i=0;i<getD_model();i++)
    {
        std::getline(ifs_biases, line);
        if(startswith(line,"#"))
        {
            i--;
            continue;
        }
        try{
            m_k_biases.push_back(std::stod(line));
        } catch (...) {
            throw std::runtime_error("Cannot convert " + line + " to a number while reading file " + biases_file);
        }
    }
    for(size_t i=0;i<getD_model();i++)
    {
        std::getline(ifs_biases, line);
        if(startswith(line,"#"))
        {
            i--;
            continue;
        }
        try{
            m_v_biases.push_back(std::stod(line));
        } catch (...) {
            throw std::runtime_error("Cannot convert " + line + " to a number while reading file " + biases_file);
        }
    }
    for(size_t i=0;i<getD_model();i++)
    {
        std::getline(ifs_biases, line);
        if(startswith(line,"#"))
        {
            i--;
            continue;
        }
        try{
            m_o_biases.push_back(std::stod(line));
        } catch (...) {
            throw std::runtime_error("Cannot convert " + line + " to a number while reading file " + biases_file);
        }
    }
    m_output_size = m_input_size;//why not the end of array m_weights?
}

// \brief calculate the output value of particular layer, with bias
std::vector<double> AttentionLayer::compute(const std::vector<double>& input) const {
    std::vector<std::vector<double>> q=transpose(reshape(input,int(m_d_model),int(m_input_size/m_d_model)));//seq_len*d_model
    std::vector<std::vector<double>> k=q;//shape=(seq_len_k,d_model)
    std::vector<std::vector<double>> v=q;//total assignment
    std::vector<std::vector<double>> buf=creatematrix(q.size(),1,1);//all one matrix
    std::vector<std::vector<double>> q_expand=hstack(q,buf);
    // std::vector<std::vector<double>> k_expand=q_expand;
    // std::vector<std::vector<double>> v_expand=q_expand;
    q=matmul(q_expand,vstack(m_q_weights,reshape(m_q_biases,1,m_q_biases.size())));
    k=matmul(q_expand,vstack(m_k_weights,reshape(m_k_biases,1,m_k_biases.size())));
    v=matmul(q_expand,vstack(m_v_weights,reshape(m_v_biases,1,m_v_biases.size())));
    std::vector<std::vector<std::vector<double>>> q_mul=reshape(q,m_num_head,true);//shape=(num_head,seq_len,d_model/num_head)
    std::vector<std::vector<std::vector<double>>> k_mul=reshape(k,m_num_head,true);//split head
    std::vector<std::vector<std::vector<double>>> v_mul=reshape(v,m_num_head,true);
    std::vector<std::vector<std::vector<double>>> qk_mul;
    for(unsigned int i=0;i<m_num_head;i++)//iterate according to heads
    {
        qk_mul.push_back(multiply_num(matmul(q_mul[i],transpose(k_mul[i])),1/sqrt(m_d_model/m_num_head)));//shape=(num_head,seq_len_q,seq_len_k)
    }
    qk_mul=softmax(qk_mul);
    std::vector<std::vector<std::vector<double>>> scaled_qkv;
    for(unsigned int i=0;i<m_num_head;i++)
    {
        //shape=(num_head, seq_len_v, depth=d_model/num_head)
        scaled_qkv.push_back(matmul(qk_mul[i],v_mul[i]));
    }
    std::vector<std::vector<double>> concate_qkv=reshape(scaled_qkv);//shape=(seq_len, d_model)
    concate_qkv=matmul(hstack(concate_qkv,buf),vstack(m_o_weights,reshape(m_o_biases,1,m_o_biases.size())));
    std::vector<double> output=concat(concate_qkv);
    return output;
}

std::vector<std::vector<double>> AttentionLayer::computeGradient(const std::vector<double>& input) const {
    /*
    * gradient of input[row*seq_len+column]={softmax((X@WQ)_head@(X@WK)_head^T/sqrt(depth))}@{partial(X@WV)_head/partial(x_ij)}+
    * {partial(softmax((X@WQ)_head@(X@WK)_head^T/sqrt(depth)))/partial(x_ij)}@{(X@WV)_head}
    * We notify the four part to be calculated as part1, part2, part3, part4. 
    * part2=partial(X)/partial(x)@WV
    * notice that split_head(X@W)==X@split_head(W), that's why we calcualate the total derivative before split head. 
    * 
    */ 
    std::vector<std::vector<double>> q=transpose(reshape(input,int(m_d_model),int(m_input_size/m_d_model)));//seq_len*d_model
    std::vector<std::vector<double>> k=q;//shape=(seq_len_k,d_model)
    std::vector<std::vector<double>> v=q;//total assignment
    std::vector<std::vector<double>> buf_1=creatematrix(q.size(),1,1);//all one matrix
    std::vector<std::vector<double>> q_expand=hstack(q,buf_1);
    // std::vector<std::vector<double>> k_expand=q_expand;
    // std::vector<std::vector<double>> v_expand=q_expand;
    std::vector<std::vector<std::vector<double>>> q_mul,k_mul,v_mul;
    std::vector<std::vector<std::vector<double>>> qk_mul(m_num_head,std::vector<std::vector<double>>(0));
    std::vector<std::vector<std::vector<double>>> qk_mul_sof;
    std::vector<std::vector<std::vector<double>>> scaled_qkv(m_num_head,std::vector<std::vector<double>>(0));
    std::vector<std::vector<double>> concate_qkv;
    std::vector<double> output;
    std::vector<std::vector<double>> output_grad(m_output_size, std::vector<double>(m_input_size, 0));
    std::vector<std::vector<double>> X;
    int seq_len=m_input_size/m_d_model;
    int depth=m_d_model/m_num_head;

    q=matmul(q_expand,vstack(m_q_weights,reshape(m_q_biases,1,m_q_biases.size())));
    q_mul=reshape(q,m_num_head,true);//shape=(num_head,seq_len,d_model/num_head)

    k=matmul(q_expand,vstack(m_k_weights,reshape(m_k_biases,1,m_k_biases.size())));
    k_mul=reshape(k,m_num_head,true);//split head

    v=matmul(q_expand,vstack(m_v_weights,reshape(m_v_biases,1,m_v_biases.size())));
    v_mul=reshape(v,m_num_head,true);

    for(unsigned int i=0;i<m_num_head;i++)//iterate according to heads
    {
        qk_mul[i]=(multiply_num(matmul(q_mul[i],transpose(k_mul[i])),1/sqrt(m_d_model/m_num_head)));//shape=(num_head,seq_len_q,seq_len_k)
    }

    qk_mul_sof=softmax(qk_mul);

    for(unsigned int i=0;i<m_num_head;i++)
    {   
        //shape=(num_head, seq_len_v, depth=d_model/num_head)
        scaled_qkv[i]=(matmul(qk_mul_sof[i],v_mul[i]));
    }

    concate_qkv=reshape(scaled_qkv);//shape=(seq_len, d_model)
    concate_qkv=matmul(hstack(concate_qkv,buf_1),vstack(m_o_weights,reshape(m_o_biases,1,m_o_biases.size())));
    output=concat(concate_qkv);
    //compute value to an END
    X=transpose(reshape(input,int(m_d_model),int(m_input_size/m_d_model)));//Notice that here is not q, because q=xw.  
    //recurrent in the matrix

    for(int row=0;row<seq_len;row++)
    {
    for(unsigned int column=0;column<m_d_model;column++)
    {
    std::vector<std::vector<std::vector<double>>> part1=qk_mul_sof;//shape=(num_head,seq_len_q,seq_len_k)
    std::vector<std::vector<std::vector<double>>> part2=reshape(matmul(one_hot_matrix(X,row,column),m_v_weights),m_num_head,true);//shape=(num_head,seq_len,depth)
    std::vector<std::vector<std::vector<double>>> part3(m_num_head);
    std::vector<std::vector<std::vector<double>>> part3_2;
    std::vector<std::vector<std::vector<double>>> part4=v_mul;//shape=(num_head, seq_len, depth)
    std::vector<std::vector<std::vector<double>>> part34(m_num_head);//shape=(num_head, seq_len, depth)
    part3_2=matmul(q_mul,reshape(matmul(transpose(m_k_weights),one_hot_matrix(transpose(X),column,row)),m_num_head,false));//Notice that this is not the regular splite head here. 
    part3_2=plus(part3_2,matmul(reshape(matmul(one_hot_matrix(X,row,column),m_q_weights),m_num_head,true),transpose(k_mul)));//shape=(num_head, seq_len, seq_len)
    part3_2=multiply_num(part3_2,1/sqrt(depth));
    std::vector<std::vector<double>> buf;
    for(unsigned int head_i=0;head_i<m_num_head;head_i++)
    {
        for(int seq_len_i=0;seq_len_i<seq_len;seq_len_i++)
        {
            buf=softmax_derivative(qk_mul[head_i][seq_len_i]);//shape=(seq_len,seq_len)
            /*
            *(XWQ)_head@(WK^T@diag(X[j][i]=1))+(diag(X[i][j]=1@WQ_head@(XWK)^T))
            */
            //buf=matmul(buf,hsplit(part3_2[head_i],seq_len_i,seq_len_i+1));
            buf=matmul(buf,reshape(part3_2[head_i][seq_len_i],seq_len,1));//shape=(seq_len,1)
            buf=transpose(buf);//shape=(1, seq_len)
            buf=matmul(buf,v_mul[head_i]);//shape=(1, depth)
            //part3[head_i].push_back(concat(buf,seq_len));
            part34[head_i].push_back(buf[0]);//final shape=(num_head, seq_len, depth)

        }
    }
    std::vector<std::vector<std::vector<double>>> result_mul=matmul(part1,part2);
    result_mul=plus(result_mul,part34);
    std::vector<std::vector<double>> result=reshape(result_mul);
    result=matmul(hstack(result,buf_1),vstack(m_o_weights,reshape(m_o_biases,1,m_o_biases.size())));
    std::vector<double> result_output=concat(result);
    
    for(unsigned int output_i=0;output_i<m_output_size;output_i++)
    {
        output_grad[output_i][column*seq_len+row]=result_output[output_i];
    }
    }
    }
    return output_grad;
}
std::ostream& AttentionLayer::showInfo(std::ostream& os) 
{
    os << "Input size: " << m_input_size << std::endl;
    os << "Output size: " << m_output_size << std::endl;
    std::string outputbuf="";
    os<<"Matrix WQ with bias: \n";
    for(size_t i=0;i<getD_model();i++)
    {
        for(size_t j=0;j<getD_model();j++)
        {
            outputbuf.append(std::to_string(getWeight_q(i,j))+"\t");
        }
        outputbuf.append("\n");
        os<<outputbuf;
        outputbuf="";
        outputbuf.append(std::to_string(getBias_q(i))+"\t");
    }
    os<<outputbuf;
    outputbuf="";
    os<<"Matrix WQ: \n";
    for(size_t i=0;i<getD_model();i++)
    {
        
        for(size_t j=0;j<getD_model();j++)
        {
            outputbuf.append(std::to_string(getWeight_k(i,j))+"\t");
        }
        outputbuf.append("\n");
        os<<outputbuf;
        outputbuf="";
        outputbuf.append(std::to_string(getBias_k(i))+"\t");
    }
    os<<outputbuf;
    outputbuf="";
    os<<"Matrix WQ: \n";
    for(size_t i=0;i<getD_model();i++)
    {
        
        for(size_t j=0;j<getD_model();j++)
        {
            outputbuf.append(std::to_string(getWeight_v(i,j))+"\t");
        }
        outputbuf.append("\n");
        os<<outputbuf;
        outputbuf="";
        outputbuf.append(std::to_string(getBias_v(i))+"\t");
    }
    os<<outputbuf;
    outputbuf="";
    os<<"Matrix WQ: \n";
    for(size_t i=0;i<getD_model();i++)
    {
        for(size_t j=0;j<getD_model();j++)
        {
            outputbuf.append(std::to_string(getWeight_o(i,j))+"\t");
        }
        outputbuf.append("\n");
        os<<outputbuf;
        outputbuf="";
        outputbuf.append(std::to_string(getBias_o(i))+"\t");
    }
    os<<outputbuf;
    outputbuf="";
    return os;
}


// \brief [wasted] creation function, @param object layers, initialized as 0
neuralNetworkCompute::neuralNetworkCompute(const std::vector<LayerType>& layer_types, const std::vector<denseLayer>& dense_layers,const std::vector<AttentionLayer>& attention_layers)
{
    m_layer_types=layer_types;
    m_dense_layers=dense_layers;
    m_attention_layers=attention_layers;
    m_layers_output.resize(m_layer_types.size());
    m_grads_tmp.resize(m_layer_types.size());
    int num_dense=0;
    int num_attentiion=0;
    for (size_t i_layer = 0; i_layer < m_layers_output.size(); ++i_layer) 
    {
        if(m_layer_types[i_layer]==LayerType::DENSE)
        {
            m_layers_output[i_layer].assign(m_dense_layers[num_dense].getOutputSize(),0);
            m_grads_tmp[i_layer].assign(m_dense_layers[num_dense].getOutputSize(),std::vector<double>(m_dense_layers[num_dense].getInputSize(),0));
            num_dense++;
        }
        else if(m_layer_types[i_layer]==LayerType::SELFATTENTION)
        {
            m_layers_output[i_layer].assign(m_attention_layers[num_attentiion].getOutputSize(),0);
            m_grads_tmp[i_layer].assign(m_attention_layers[num_attentiion].getOutputSize(),std::vector<double>(m_attention_layers[num_attentiion].getInputSize(),0));
            num_attentiion++;
        }
    }
}
// \breif [wasted]
bool neuralNetworkCompute::addLayerTypes(const std::vector<LayerType>& layertypes)
{
    m_layer_types=layertypes;
    return true;
}
// \brief prepare the space for each layer with initialization of 0
bool neuralNetworkCompute::addDenseLayer(const denseLayer& layer) {
    if(m_layer_types.empty())
    {
        m_layer_types.push_back(LayerType::DENSE);
        m_dense_layers.push_back(layer);
        m_layers_output.push_back(std::vector<double>(layer.getOutputSize()));
        m_grads_tmp.push_back(std::vector<std::vector<double>>(layer.getOutputSize(), std::vector<double>(layer.getInputSize(), 0)));
        return true;
    }else if(m_layer_types.back()==LayerType::DENSE&&m_dense_layers.back().getOutputSize() == layer.getInputSize())
    {
        // otherwise, we need to check if the output of last layer in m_dense_layers matches the input of layer to be added
        m_layer_types.push_back(LayerType::DENSE);
        m_dense_layers.push_back(layer);
        m_layers_output.push_back(std::vector<double>(layer.getOutputSize()));
        m_grads_tmp.push_back(std::vector<std::vector<double>>(layer.getOutputSize(), std::vector<double>(layer.getInputSize(), 0)));
        return true;
    }else if(m_layer_types.back()==LayerType::SELFATTENTION&&m_attention_layers.back().getOutputSize()==layer.getInputSize())
    {
        m_layer_types.push_back(LayerType::DENSE);
        m_dense_layers.push_back(layer);
        m_layers_output.push_back(std::vector<double>(layer.getOutputSize()));
        m_grads_tmp.push_back(std::vector<std::vector<double>>(layer.getOutputSize(), std::vector<double>(layer.getInputSize(), 0)));
        return true;
    }else 
    {
        std::cout<<"not standard type"<<std::endl;
        return false;
    }
}
// \brief prepare the space for each layer with initialization of 0
bool neuralNetworkCompute::addAttentionLayer(const AttentionLayer& layer) {
    if(m_layer_types.empty())
    {
        m_layer_types.push_back(LayerType::SELFATTENTION);
        m_attention_layers.push_back(layer);
        m_layers_output.push_back(std::vector<double>(layer.getOutputSize()));
        m_grads_tmp.push_back(std::vector<std::vector<double>>(layer.getOutputSize(), std::vector<double>(layer.getInputSize(), 0)));
        return true;
    }else if(m_layer_types.back()==LayerType::DENSE&&m_dense_layers.back().getOutputSize() == layer.getInputSize())
    {
        // otherwise, we need to check if the output of last layer in m_dense_layers matches the input of layer to be added
        m_layer_types.push_back(LayerType::SELFATTENTION);
        m_attention_layers.push_back(layer);
        m_layers_output.push_back(std::vector<double>(layer.getOutputSize()));
        m_grads_tmp.push_back(std::vector<std::vector<double>>(layer.getOutputSize(), std::vector<double>(layer.getInputSize(), 0)));
        return true;
    }else if(m_layer_types.back()==LayerType::SELFATTENTION&&m_attention_layers.back().getOutputSize()==layer.getInputSize())
    {
        m_layer_types.push_back(LayerType::SELFATTENTION);
        m_attention_layers.push_back(layer);
        m_layers_output.push_back(std::vector<double>(layer.getOutputSize()));
        m_grads_tmp.push_back(std::vector<std::vector<double>>(layer.getOutputSize(), std::vector<double>(layer.getInputSize(), 0)));
        return true;
    }else 
    {
        return false;
    }
}
// \brief calculate the gradient from front to the end using the chain role
void neuralNetworkCompute::compute() {
    if (m_layer_types.empty()) {
        std::cout<<"No layer contained in the model. "<<std::endl;
        return;
    }
    int num_dense=0;
    int num_attention=0;
    std::vector<double> last_output=m_input;
    for(unsigned int i_layer=0;i_layer<m_layer_types.size();i_layer++)
    {
        if(m_layer_types[i_layer]==LayerType::DENSE)
        {
            m_dense_layers[num_dense].compute(last_output, m_layers_output[i_layer]);
            last_output=m_layers_output[i_layer];
            num_dense++;
        }else if(m_layer_types[i_layer]==LayerType::SELFATTENTION)
        {
            m_layers_output[i_layer]=m_attention_layers[num_attention].compute(last_output);
            last_output=m_layers_output[i_layer];
            num_attention++;
        }else
        {
            std::cout<<"This kind of layer is not supported. "<<std::endl;
            return;
        }
        
    }
    /*
    m_layers_output[0] = m_dense_layers[0].compute(m_input);
    for (size_t i_layer = 1; i_layer < m_dense_layers.size(); ++i_layer) {
        m_layers_output[i_layer] = m_dense_layers[i_layer].compute(m_layers_output[i_layer - 1]);
    }
    */
    // gradients of each layer
//     std::vector<std::vector<std::vector<double>>> grads(m_dense_layers.size());
    last_output=m_input;
    num_dense=0;
    num_attention=0;
    for(unsigned int i_layer=0;i_layer<m_layer_types.size();i_layer++)
    {
        if(m_layer_types[i_layer]==LayerType::DENSE)
        {
            m_dense_layers[num_dense].computeGradient(last_output, m_grads_tmp[i_layer]);
            last_output=m_layers_output[i_layer];
            num_dense++;
        }else if(m_layer_types[i_layer]==LayerType::SELFATTENTION)
        {
            m_grads_tmp[i_layer]=m_attention_layers[num_attention].computeGradient(last_output);
            last_output=m_layers_output[i_layer];
            num_attention++;
        }else
        {
            std::cout<<"This kind of layer is not supported. "<<std::endl;
            return;
        }
        
    }
    /*
    m_grads_tmp[0] = m_dense_layers[0].computeGradient(m_input);
    for (size_t i_layer = 1; i_layer < m_dense_layers.size(); ++i_layer) {
        m_grads_tmp[i_layer] = m_dense_layers[i_layer].computeGradient(m_layers_output[i_layer - 1]);
    }
    */
    // chain rule
    if (m_layer_types.size() > 1) {
        //std::vector<std::vector<double>> m_chained_grad=diag_matrix(std::vector<double>(m_input.size(),1));
        //Why calculate the reverse order of m_grade_tmp? what is m_grades_tmp[0]?
        m_chained_grad = m_grads_tmp[0];
        for(size_t i_layer=1;i_layer<m_layer_types.size();i_layer++)
        {
            m_chained_grad=multiply_matrix(m_grads_tmp[i_layer],m_chained_grad);
        }
        /*
        m_chained_grad = multiply_matrix(m_grads_tmp[1], m_grads_tmp[0]);
        for (size_t i_layer = 2; i_layer < m_dense_layers.size(); ++i_layer) {
            m_chained_grad = multiply_matrix(m_grads_tmp[i_layer], m_chained_grad);
        }
        */
    } else {
        m_chained_grad = m_grads_tmp[0];
    }
}
}
