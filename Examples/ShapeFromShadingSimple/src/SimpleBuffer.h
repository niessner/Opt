#ifndef SimpleBuffer_h
#define SimpleBuffer_h
#include <string>
class SimpleBuffer {
public:
    enum DataType { FLOAT = 0, UCHAR = 1 };
    static size_t datatypeToSize(SimpleBuffer::DataType dt) {
        return (dt == DataType::FLOAT) ? sizeof(float) : sizeof(unsigned char);
    }
protected:
    bool        m_onGPU;
    int         m_width;
    int         m_height;
    int         m_channelCount;
    DataType    m_dataType;
    void*       m_data;

    SimpleBuffer(std::string filename, bool onGPU);
    SimpleBuffer(const SimpleBuffer& other, bool onGPU);

public:
    void save(std::string filename);
    ~SimpleBuffer();
};

#endif