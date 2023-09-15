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
#ifndef INC_BVH_VIS_BVH_HPP
#define INC_BVH_VIS_BVH_HPP

#ifdef BVH_USE_VTK

#include "vtk_helpers.hpp"
#include "vis_kdop.hpp"
#include <vtkPolyData.h>
#include <vtkAppendPolyData.h>
#include "../tree.hpp"
#include "../tree_iterator.hpp"
#include "../iterators/level_iterator.hpp"
#include <ostream>

namespace bvh
{
  namespace vis
  {
    namespace detail
    {
      template< typename Range >
      vtk_unique_ptr< vtkPolyData >
      node_range_polydata( Range &&_range )
      {
        auto poly_list = make_vtkobj< vtkAppendPolyData >();
        for ( auto &&node : _range )
        {
          poly_list->AddInputData( kdop_polydata( node.kdop() ).release() );
        }
        
        poly_list->Update();
  
        // VTK ownership semantics are strange, need to do a deep copy to retain
        // ownership of the polydata...
        auto ret = make_vtkobj< vtkPolyData >();
        ret->DeepCopy( poly_list->GetOutput() );
        
        return std::move( ret );
      }
    }
    
    template< typename T, typename KDop, typename NodeData >
    vtk_unique_ptr< vtkPolyData >
    bvh_polydata( const bvh_tree< T, KDop, NodeData > &_tree )
    {
      return detail::node_range_polydata( make_preorder_traverse( _tree ) );
    };

    template< typename T, typename KDop, typename NodeData >
    vtk_unique_ptr< vtkPolyData >
    bvh_level_polydata( const bvh_tree< T, KDop, NodeData > &_tree,
                        int _level )
    {
      return detail::node_range_polydata( max_level_traverse( _tree, _level ) );
    };

    template< typename T, typename KDop, typename NodeData >
    vtk_unique_ptr< vtkPolyData >
    bvh_leaf_polydata( const bvh_tree< T, KDop, NodeData > &_tree )
    {
      return detail::node_range_polydata( leaf_traverse( _tree ) );
    };

    template< typename T, typename KDop, typename NodeData >
    std::ostream &
    write_bvh( std::ostream &_os, const bvh_tree< T, KDop, NodeData > &_tree )
    {
      auto pd = bvh_polydata( _tree );
    
      auto writer = make_vtkobj< vtkXMLPolyDataWriter >();
      writer->SetInputData( pd.get() );
      writer->WriteToOutputStringOn();
      writer->Write();
    
      _os << writer->GetOutputString();
    
      return _os;
    };

    template< typename T, typename KDop, typename NodeData >
    std::ostream &
    write_bvh_level( std::ostream &_os, const bvh_tree< T, KDop, NodeData > &_tree,
                     int _level )
    {
      auto pd = bvh_level_polydata( _tree, _level );
    
      auto writer = make_vtkobj< vtkXMLPolyDataWriter >();
      writer->SetInputData( pd.get() );
      writer->WriteToOutputStringOn();
      writer->Write();
    
      _os << writer->GetOutputString();
    
      return _os;
    };

    template< typename T, typename KDop, typename NodeData >
    std::ostream &
    write_bvh_leaves( std::ostream &_os, const bvh_tree< T, KDop, NodeData > &_tree )
    {
      auto pd = bvh_leaf_polydata( _tree );
    
      auto writer = make_vtkobj< vtkXMLPolyDataWriter >();
      writer->SetInputData( pd.get() );
      writer->WriteToOutputStringOn();
      writer->Write();
    
      _os << writer->GetOutputString();
    
      return _os;
    };
  }
}

#endif  // BVH_USE_VTK

#endif  // INC_BVH_VIS_BVH_HPP
