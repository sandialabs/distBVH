/*
 * distBVH 1.0
 *
 * Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC
 * (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
 * Government retains certain rights in this software.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef INC_BVH_VIS_KDOP_HPP
#define INC_BVH_VIS_KDOP_HPP

#ifdef BVH_USE_VTK

#include "vtk_helpers.hpp"
#include "../kdop.hpp"
#include <ostream>
#include <vector>
#include <vtkPolyData.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkPlanes.h>
#include <vtkHull.h>
#include <vtkAppendPolyData.h>

namespace bvh
{
  namespace vis
  {
    template< typename T, int K, typename Derived >
    vtk_unique_ptr< vtkPolyData >
    kdop_polydata( const kdop_base< T, K, Derived > &_kdop )
    {
      auto normals = make_vtkobj< vtkDoubleArray >();
      auto points = make_vtkobj< vtkPoints >();
  
      normals->SetNumberOfComponents( 3 );
      normals->SetNumberOfTuples( K );
  
      points->SetNumberOfPoints( K );
  
      for ( int i = 0; i < K / 2; ++i )
      {
        auto pt1 = _kdop.extents[i].max * Derived::normals()[i];
        auto pt2 = _kdop.extents[i].min * Derived::normals()[i];
        auto norm1 = Derived::normals()[i];
        auto norm2 = Derived::normals()[i + K / 2];
    
        normals->SetTuple3( i, norm1.x(), norm1.y(), norm1.z() );
        points->SetPoint( i, pt1.x(), pt1.y(), pt1.z() );
        normals->SetTuple3( i + K / 2, norm2.x(), norm2.y(), norm2.z() );
        points->SetPoint( i + K / 2, pt2.x(), pt2.y(), pt2.z() );
      }
  
      auto planes = make_vtkobj< vtkPlanes >();
  
      planes->SetPoints( points.get() );
      planes->SetNormals( normals.get() );
  
      auto hull = make_vtkobj< vtkHull >();
      hull->SetPlanes( planes.get() );
  
      auto pd = make_vtkobj< vtkPolyData >();
      hull->GenerateHull( pd.get(), _kdop.extents[0].min, _kdop.extents[0].max,
                          _kdop.extents[1].min, _kdop.extents[1].max,
                          _kdop.extents[2].min, _kdop.extents[2].max );
      
      return std::move( pd );
    };
    
    
    template< typename T, int K, typename Derived >
    std::ostream &
    write_kdop( std::ostream &_os, const kdop_base< T, K, Derived > &_kdop )
    {
      auto pd = kdop_polydata( _kdop );
      
      auto writer = make_vtkobj< vtkXMLPolyDataWriter >();
      writer->SetInputData( pd.get() );
      writer->WriteToOutputStringOn();
      writer->Write();
      
      _os << writer->GetOutputString();
      
      return _os;
    };
  
    
    template< typename Kdop >
    std::ostream &
    write_kdop_list( std::ostream &_os, const std::vector< Kdop > &_kdops )
    {
      if ( _kdops.empty() )
        return _os;
      
      auto poly_list = make_vtkobj< vtkAppendPolyData >();
      for ( auto &&kdop : _kdops )
      {
        auto pd = kdop_polydata( kdop );
        poly_list->AddInputData( pd.release() );
      }
  
      poly_list->Update();
      auto pd = make_vtkobj< vtkPolyData >();
      pd->DeepCopy( poly_list->GetOutput() );
      
      auto writer = make_vtkobj< vtkXMLPolyDataWriter >();
      writer->SetInputData( pd.get() );
      writer->WriteToOutputStringOn();
      writer->Write();
    
      _os << writer->GetOutputString();
    
      return _os;
    };
  }
}

#endif

#endif  // INC_BVH_VIS_KDOP_HPP
