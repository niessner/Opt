#pragma once
#include <vector>
#include <algorithm>

/**
Uses SoA, fairly efficient for small # of parameters.
If parameter count could be large, consider better approaches
*/
class NamedParameters {
public:
    void** data() const {
        return (void**)m_data.data();
    }
    void set(const std::string& name, void* data) {
        auto location = std::find(m_names.begin(), m_names.end(), name);
        if (location == m_names.end()) {
            m_names.push_back(name);
            m_data.push_back(data);
        }
        else {
            *location = name;
        }
    }
    std::vector<std::string> names() const {
        return m_names;
    }

protected:
    std::vector<void*> m_data;
    std::vector<std::string> m_names;
};