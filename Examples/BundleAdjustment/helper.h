
namespace helper
{
    inline int countBits(UINT64 v)
    {
        #ifdef _WIN32        
        return (int)__popcnt64(v);
        #else
        return __builtin_popcountll(v);
        #endif
    }
    inline void splatPoint(Bitmap &bmp, int x, int y, vec4uc color)
    {
        int radius = 1;
        for (int xOffset = -radius; xOffset <= radius; xOffset++)
            for (int yOffset = -radius; yOffset <= radius; yOffset++)
            {
                if (bmp.isValidCoordinate(x + xOffset, y + yOffset))
                    bmp(x + xOffset, y + yOffset) = color;
            }
    }

    inline void splatPoint(Bitmap &bmp, vec2i coord, vec4uc color)
    {
        splatPoint(bmp, coord.x, coord.y, color);
    }

    inline vec4uc randomMatchColor()
    {
        return vec4uc((BYTE)util::randomInteger(64, 255),
                      (BYTE)util::randomInteger(64, 255),
                      (BYTE)util::randomInteger(64, 255), 255);
    }
}
