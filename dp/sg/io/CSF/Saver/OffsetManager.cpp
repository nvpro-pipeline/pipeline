// Copyright NVIDIA Corporation 2002-2012
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


#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "OffsetManager.h"


#if defined(_MSC_VER ) && _MSC_VER >= 1500  && (defined(__amd64__) || defined(__x86_64__) || defined(_M_X64) || defined(__AMD64__))
  #define xftell(f)         _ftelli64(f)
  #define xfseek(f,pos,key) _fseeki64(f,pos,key)
#else
  #define xftell(f)         ftell(f)
  #define xfseek(f,pos,key) fseek(f,(long)pos,key)
#endif

struct CSFOffsetMgr 
{
  struct Entry
  {
    CSFoffset offset;
    CSFoffset location;
  };

  FILE*               m_file;
  std::vector<Entry>  m_offsetLocations;
  size_t              m_current;
  int                 m_overflow;


  CSFOffsetMgr(FILE* file) : m_current(0), m_file(file), m_overflow(0)
  {

  }

  size_t store(const void* data, size_t dataSize)
  {
    if (m_overflow){
      return 0;
    }

    size_t last = m_current;
    fwrite(data, dataSize, 1, m_file);

    m_current += dataSize;

    m_overflow |= m_current < last;

    return last;
  }

  size_t store(size_t location, const void* data, size_t dataSize)
  {
    if (m_overflow){
      return 0;
    }

    size_t last = m_current;
    fwrite(data, dataSize, 1, m_file);

    m_current += dataSize;

    m_overflow |= m_current < last;

    Entry entry = {last, location};
    m_offsetLocations.push_back(entry);

    return last;
  }

  void finalize(size_t tableCountLocation, size_t tableLocation)
  {
    if (m_overflow){
      return;
    }

    xfseek (m_file, tableCountLocation, SEEK_SET);
    int num = int(m_offsetLocations.size());
    fwrite(&num, sizeof(int), 1, m_file);

    CSFoffset offset = (CSFoffset)m_current;
    xfseek (m_file, tableLocation, SEEK_SET);
    fwrite(&offset, sizeof(CSFoffset), 1, m_file);

    for (size_t i = 0; i < m_offsetLocations.size(); i++)
    {
      xfseek (m_file, m_offsetLocations[i].location, SEEK_SET);
      fwrite(&m_offsetLocations[i].offset, sizeof(CSFoffset), 1, m_file);
    }

    // dump table
    xfseek (m_file, 0, SEEK_END);
    for (size_t i = 0; i < m_offsetLocations.size(); i++)
    {
      fwrite(&m_offsetLocations[i].location, sizeof(CSFoffset), 1, m_file);
    }
  }

  bool hadOverflow()
  {
    return !!m_overflow;
  }
};

int CSFile_save(const CSFile* csf, const char* filename)
{
  FILE* file;
  if (fopen_s(&file,filename,"wb")){
    return CADSCENEFILE_ERROR_NOFILE;
  }

  CSFOffsetMgr mgr(file);

  CSFile dump = *csf;
  dump.version = CADSCENEFILE_VERSION;
  dump.magic   = CADSCENEFILE_MAGIC;
  // dump main part as is
  mgr.store(&dump,sizeof(CSFile));

  // iterate the objects

  {
    size_t geomOFFSET = mgr.store(offsetof(CSFile,geometriesOFFSET), 
      csf->geometries, sizeof(CSFGeometry) * csf->numGeometries);

    for (int i = 0; i < csf->numGeometries; i++,geomOFFSET+=sizeof(CSFGeometry))
    {
      const CSFGeometry* geo = csf->geometries + i;
      if (geo->vertex && geo->numVertices)
      {
        mgr.store( geomOFFSET + offsetof(CSFGeometry,vertexOFFSET),
          geo->vertex, sizeof(float) * 3 * geo->numVertices);
      }
      if (geo->normal && geo->numVertices)
      {
        mgr.store( geomOFFSET + offsetof(CSFGeometry,normalOFFSET),
          geo->normal, sizeof(float) * 3 * geo->numVertices);
      }
      if (geo->tex && geo->numVertices)
      {
        mgr.store( geomOFFSET + offsetof(CSFGeometry,texOFFSET),
          geo->tex, sizeof(float) * 2 * geo->numVertices);
      }
      if (geo->indexSolid && geo->numIndexSolid)
      {
        mgr.store( geomOFFSET + offsetof(CSFGeometry,indexSolidOFFSET),
          geo->indexSolid, sizeof(int) * geo->numIndexSolid);
      }
      if (geo->indexWire && geo->numIndexWire)
      {
        mgr.store( geomOFFSET + offsetof(CSFGeometry,indexWireOFFSET),
          geo->indexWire, sizeof(int)  * geo->numIndexWire);
      }
      if (geo->parts && geo->numParts)
      {
        mgr.store( geomOFFSET + offsetof(CSFGeometry,partsOFFSET),
          geo->parts, sizeof(CSFGeometryPart)  * geo->numParts);
      }
    }
  }


  {
    size_t matOFFSET = mgr.store(offsetof(CSFile,materialsOFFSET), 
      csf->materials, sizeof(CSFMaterial) * csf->numMaterials);

    for (int i = 0; i < csf->numMaterials; i++, matOFFSET+= sizeof(CSFMaterial))
    {
      const CSFMaterial* mat = csf->materials + i;
      if (mat->bytes && mat->numBytes)
      {
        mgr.store(matOFFSET + offsetof(CSFMaterial,bytesOFFSET),
          mat->bytes, sizeof(unsigned char) * mat->numBytes);
      }
    }
  }

  {
    size_t nodeOFFSET = mgr.store(offsetof(CSFile,nodesOFFSET), 
      csf->nodes, sizeof(CSFNode) * csf->numNodes);

    for (int i = 0; i < csf->numNodes; i++, nodeOFFSET+=sizeof(CSFNode))
    {
      const CSFNode* node = csf->nodes + i;
      if (node->parts && node->numParts)
      {
        mgr.store(nodeOFFSET + offsetof(CSFNode,partsOFFSET),
          node->parts, sizeof(CSFNodePart) * node->numParts);
      }
      if (node->children && node->numChildren)
      {
        mgr.store(nodeOFFSET + offsetof(CSFNode,childrenOFFSET),
          node->children, sizeof(int) * node->numChildren);
      }
    }
  }

  mgr.finalize(offsetof(CSFile,numPointers),offsetof(CSFile,pointersOFFSET));

  fclose(file);

  return mgr.hadOverflow() ? CADSCENEFILE_ERROR_FILEOVERSIZED : CADSCENEFILE_NOERROR;
}
