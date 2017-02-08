#pragma once
#include <vector>
#include <algorithm>
#include "OptImage.h"
#include "OptGraph.h"


/**
Uses SoA, fairly efficient for small # of parameters.
If parameter count could be large, consider better approaches
*/
class NamedParameters {
public:
    struct Parameter {
        std::string name = "";
        void* ptr = nullptr;
        std::shared_ptr<OptImage> im = nullptr;
        Parameter() {}
        Parameter(std::string _name, void* _ptr, std::shared_ptr<OptImage> _im) : name(_name), ptr(_ptr), im(_im) {}
    };

    std::vector<Parameter> unknownParameters() const {
        std::vector<Parameter> params;
        for (auto p : m_parameters) {
            if (p.im && p.im->isUnknown()) {
                params.push_back(p);
            }
        }
        return params;
    }

    std::vector<void*> data() const {
        std::vector<void*> d(m_parameters.size());
        std::transform(m_parameters.begin(), m_parameters.end(), d.begin(), [](const Parameter& p){return p.ptr; });
        return d;
    }

    void get(const std::string& name, Parameter& param) const {
        auto location = std::find_if(m_parameters.begin(), m_parameters.end(), [name](const Parameter& p){ return p.name == name; });
        param = *location;
    }

    void set(const std::string& name, void* data) {
        auto location = std::find_if(m_parameters.begin(), m_parameters.end(), [name](const Parameter& p){ return p.name == name; });
        if (location == m_parameters.end()) {
            m_parameters.push_back(Parameter(name, data, std::shared_ptr<OptImage>()));
        } else {
            auto index = std::distance(m_parameters.begin(), location);
            location->ptr = data;
            location->im = nullptr;
        }
    }
    void set(const std::string& name, std::shared_ptr<OptImage> im) {
        auto location = std::find_if(m_parameters.begin(), m_parameters.end(), [name](const Parameter& p){ return p.name == name; });
        if (location == m_parameters.end()) {
            m_parameters.push_back(Parameter(name, im->data(), im));
        }
        else {
            auto index = std::distance(m_parameters.begin(), location);
            location->ptr = im->data();
            location->im = im;
        }
    }
    std::vector<std::string> names() const {
        std::vector<std::string> d(m_parameters.size());
        std::transform(m_parameters.begin(), m_parameters.end(), d.begin(), [](const Parameter& p){return p.name; });
        return d;
    }

protected:
    std::vector<Parameter> m_parameters;

};