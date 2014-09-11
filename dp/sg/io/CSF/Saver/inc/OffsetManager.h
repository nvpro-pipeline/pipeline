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


#pragma once


extern "C" 
{

#if defined(WIN32)
  #define CSFAPI __declspec(dllexport)
#else
  #define CSFAPI
#endif

enum {
  CADSCENEFILE_VERSION = 2,

  CADSCENEFILE_NOERROR = 0,
  CADSCENEFILE_ERROR_NOFILE = 1,
  CADSCENEFILE_ERROR_VERSION = 2,
  CADSCENEFILE_ERROR_FILEOVERSIZED = 3,
  CADSCENEFILE_MAGIC = 1567262451,
};

typedef unsigned long long CSFoffset;

struct CSFMaterial {
  char    name[128];
  float   color[4];
  int     type;
  int     numBytes;
  union {
    CSFoffset           bytesOFFSET;
    unsigned char*      bytes;
  };
};

struct CSFGeometryPart
{
  int vertex;
  int indexSolid;
  int indexWire;
};

struct CSFGeometry
{
  float matrix[16];
  int                 numParts;
  int                 numVertices;
  int                 numIndexSolid;
  int                 numIndexWire;

  union
  {
    CSFoffset         vertexOFFSET;
    float*            vertex;
  };

  union
  {
    CSFoffset         normalOFFSET;
    float*            normal;
  };

  union
  {
    CSFoffset         texOFFSET;
    float*            tex;
  };

  union 
  {
    CSFoffset         indexSolidOFFSET;
    unsigned int*     indexSolid;
  };

  union 
  {
    CSFoffset         indexWireOFFSET;
    unsigned int*     indexWire;
  };

  union 
  {
    CSFoffset         partsOFFSET;
    CSFGeometryPart*  parts;
  };
};

struct CSFNodePart 
{
  int                 active;
  int                 materialIDX;
  float               linewidth;
};

struct CSFNode
{
  float               objectTM[16];
  float               worldTM[16];
  int                 geometryIDX;
  int                 numParts;
  int                 numChildren;
  union 
  {
    CSFoffset         partsOFFSET;
    CSFNodePart*      parts;
  };
  union 
  {
    CSFoffset         childrenOFFSET;
    int*              children;
  };
};


struct CSFile 
{
  int                   magic;
  int                   version;
  int                   hasUniqueNodes;
  int                   numPointers;
  int                   numGeometries;
  int                   numMaterials;
  int                   numNodes;
  int                   rootIDX;

  union 
  {
    CSFoffset           pointersOFFSET;
    CSFoffset*          pointers;
  };

  union 
  {
    CSFoffset           geometriesOFFSET;
    CSFGeometry*        geometries;
  };

  union {
    CSFoffset           materialsOFFSET;
    CSFMaterial*        materials;
  };

  union 
  {
    CSFoffset           nodesOFFSET;
    CSFNode*            nodes;
  };
};

  CSFAPI int     CSFile_save  (const CSFile* file, const char* filename);
};
