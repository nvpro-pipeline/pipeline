// Copyright NVIDIA Corporation 2010
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#pragma once
/** @file */

#include <dp/sg/core/Texture.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {


      /** \brief Interface for textures used as storage for SceniX. 
       *  \sa TextureHost, TextureGL
      **/
      class TextureFile : public Texture
      {
      public:
        /*! \brief Create a Texture object referencing a file. If this function is being called multiple times with the
                   same filename it'll return the same TextureFile object as long as there was at least one reference
                   to an object for the same file. The textureTarget is not considered for this cache. It is illegal
                   to specify different textureTargets for the same filename.
            \param filename Filename of the image for the given texture
            \param textureTarget TextureTarget of the texture
        **/
        DP_SG_CORE_API static TextureFileSharedPtr create( const std::string& filename, TextureTarget textureTarget = TT_UNSPECIFIED_TEXTURE_TARGET );

        DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

      public:
        const std::string& getFilename() const;

        /*! \brief Get the hash key of this Object.
         *  \return The hash key of this Object.
         *  \remarks If the hash key is not valid, the virtual function feedHashGenerator() is called to
         *  recursively determine a hash string, using a HashGenerator, which then is converted into
         *  a HashKey.
         *  \sa feedHashGenerator */
        dp::util::HashKey getHashKey() const;

        /*! \brief Tests whether this Texture is equivalent to another Texture.
          * \param p Pointer to the Texture to test for equivalence with this Texture object.
          * \param deepCompare The function performs a deep-compare instead of a shallow compare if this is \c true.
          * \return The function returns \c true if the Texture pointed to by \a p is detected to be equivalent
          * to this Texture object. */
        DP_SG_CORE_API virtual bool isEquivalent( TextureSharedPtr const& texture, bool deepCompare ) const;

      protected:
        DP_SG_CORE_API TextureFile( const std::string& filename, TextureTarget textureTarget = TT_UNSPECIFIED_TEXTURE_TARGET );

        /*! \brief Feed the data of this object into the provied HashGenerator.
         *  \param hg The HashGenerator to update with the data of this object.
         *  \sa getHashKey */
        DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        /*! \brief Invalidate the HashKey.
         *  \sa getHashKey, feedHashGenerator */
        void invalidateHashKey();

    #if !defined(NDEBUG)
        bool isHashKeyValid() const;
    #endif

      private:
        friend class TextureFileCache;

        std::string               m_filename;
      };

      inline const std::string& TextureFile::getFilename() const
      {
        return m_filename;
      }

    } // namespace core
  } // namespace sg
} // namespace dp

