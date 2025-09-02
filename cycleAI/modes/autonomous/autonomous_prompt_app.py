"""
Streamlit Interface for Autonomous Prompt Generation
"""

import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from autonomous_prompt_generator import AutonomousPromptGenerator
from datetime import datetime
import time

def main():
    st.set_page_config(
        page_title="🤖 Autonomous Prompt Generator", 
        page_icon="🤖", 
        layout="wide"
    )
    
    st.title("🤖 Autonomous Baseball Prompt Generator")
    st.markdown("*Automatically generate, test, and optimize baseball queries*")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        st.session_state.generator = AutonomousPromptGenerator()
        st.session_state.generator.load_prompt_library()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Generation Controls")
        
        batch_size = st.slider("Prompts per Batch", 5, 50, 20)
        test_limit = st.slider("Tests per Batch", 3, 15, 8)
        cycles = st.slider("Cycles", 1, 10, 3)
        
        st.header("📊 Current Library Stats")
        total_prompts = len(st.session_state.generator.generated_prompts)
        tested_prompts = len([p for p in st.session_state.generator.generated_prompts if p.tested])
        successful_prompts = len([p for p in st.session_state.generator.generated_prompts if p.success])
        
        st.metric("Total Prompts", total_prompts)
        st.metric("Tested", tested_prompts)
        st.metric("Successful", successful_prompts)
        
        if tested_prompts > 0:
            success_rate = successful_prompts / tested_prompts * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Generate", "📊 Analytics", "🎯 Best Prompts", 
        "📚 Library", "🔬 Live Testing"
    ])
    
    with tab1:
        st.header("🚀 Autonomous Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎲 Generate Single Batch", type="primary"):
                with st.spinner("Generating and testing prompts..."):
                    results = st.session_state.generator.generate_and_test_autonomous_batch(
                        batch_size=batch_size,
                        test_limit=test_limit
                    )
                    
                    st.success("Batch complete!")
                    
                    # Display results
                    test_results = results['test_results']
                    st.write(f"**Generated**: {results['generation_results']['prompts_generated']} prompts")
                    st.write(f"**Tested**: {test_results['total_tested']} prompts")
                    st.write(f"**Successful**: {test_results['successful']} prompts")
                    st.write(f"**Success Rate**: {test_results['successful']/max(test_results['total_tested'],1)*100:.1f}%")
                    
                    # Show recommendations
                    if results['recommendations']:
                        st.subheader("💡 Recommendations")
                        for rec in results['recommendations']:
                            st.write(f"• {rec}")
        
        with col2:
            if st.button("🔄 Run Continuous Mode", type="secondary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.empty()
                
                with st.spinner("Running continuous generation..."):
                    # Run continuous mode with progress updates
                    all_results = []
                    
                    for cycle in range(cycles):
                        status_text.text(f"Running cycle {cycle + 1}/{cycles}...")
                        progress_bar.progress((cycle) / cycles)
                        
                        cycle_results = st.session_state.generator.generate_and_test_autonomous_batch(
                            batch_size=batch_size,
                            test_limit=test_limit
                        )
                        
                        all_results.append(cycle_results)
                        
                        # Update results display
                        with results_container.container():
                            st.write(f"**Cycle {cycle + 1} Results:**")
                            test_res = cycle_results['test_results']
                            st.write(f"Generated: {cycle_results['generation_results']['prompts_generated']}")
                            st.write(f"Success Rate: {test_res['successful']}/{test_res['total_tested']}")
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Continuous mode complete!")
                    
                    # Final summary
                    total_generated = sum(r['generation_results']['prompts_generated'] for r in all_results)
                    total_tested = sum(r['test_results']['total_tested'] for r in all_results)
                    total_successful = sum(r['test_results']['successful'] for r in all_results)
                    
                    st.success(f"Generated {total_generated} prompts, tested {total_tested}, {total_successful} successful!")
    
    with tab2:
        st.header("📊 Performance Analytics")
        
        if st.session_state.generator.generated_prompts:
            # Create performance dataframe
            prompts_data = []
            for p in st.session_state.generator.generated_prompts:
                if p.tested:
                    prompts_data.append({
                        'category': p.category,
                        'complexity': p.complexity,
                        'quality_score': p.response_quality,
                        'response_length': p.response_length,
                        'success': p.success,
                        'timestamp': p.timestamp
                    })
            
            if prompts_data:
                df = pd.DataFrame(prompts_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Category performance chart
                    category_stats = df.groupby('category').agg({
                        'success': ['count', 'sum'],
                        'quality_score': 'mean'
                    }).round(2)
                    
                    category_stats.columns = ['total', 'successful', 'avg_quality']
                    category_stats['success_rate'] = (category_stats['successful'] / category_stats['total'] * 100).round(1)
                    
                    fig1 = px.bar(
                        category_stats.reset_index(),
                        x='category',
                        y='success_rate',
                        title="Success Rate by Category",
                        color='avg_quality',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Quality distribution
                    fig2 = px.histogram(
                        df,
                        x='quality_score',
                        nbins=20,
                        title="Quality Score Distribution",
                        color='complexity'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Complexity analysis
                complexity_stats = df.groupby('complexity').agg({
                    'success': ['count', 'sum'],
                    'quality_score': 'mean',
                    'response_length': 'mean'
                }).round(2)
                
                st.subheader("📈 Performance by Complexity")
                st.dataframe(complexity_stats, use_container_width=True)
                
            else:
                st.info("No tested prompts yet. Run some generation cycles to see analytics.")
        else:
            st.info("No prompts generated yet. Use the Generate tab to create prompts.")
    
    with tab3:
        st.header("🎯 Best Performing Prompts")
        
        best_prompts = st.session_state.generator.get_best_prompts(20)
        
        if best_prompts:
            for i, prompt in enumerate(best_prompts, 1):
                with st.expander(f"{i}. {prompt['prompt'][:80]}... (Quality: {prompt['quality_score']:.2f})"):
                    st.write(f"**Full Prompt**: {prompt['prompt']}")
                    st.write(f"**Category**: {prompt['category']}")
                    st.write(f"**Complexity**: {prompt['complexity']}")
                    st.write(f"**Quality Score**: {prompt['quality_score']:.2f}")
                    st.write(f"**Response Length**: {prompt['response_length']} characters")
                    
                    # Test this prompt live
                    if st.button(f"🧪 Test This Prompt", key=f"test_{i}"):
                        with st.spinner("Testing prompt..."):
                            try:
                                response = st.session_state.generator.agent.process_query(prompt['prompt'])
                                st.write("**Response:**")
                                st.write(response)
                            except Exception as e:
                                st.error(f"Error testing prompt: {e}")
        else:
            st.info("No successful prompts yet. Generate and test some prompts first!")
    
    with tab4:
        st.header("📚 Prompt Library")
        
        # Library overview
        col1, col2, col3 = st.columns(3)
        
        all_prompts = st.session_state.generator.generated_prompts
        
        with col1:
            st.metric("Total Prompts", len(all_prompts))
        
        with col2:
            categories = set(p.category for p in all_prompts)
            st.metric("Categories", len(categories))
        
        with col3:
            templates = len(st.session_state.generator.prompt_templates)
            st.metric("Templates", templates)
        
        # Category breakdown
        if all_prompts:
            category_counts = {}
            for p in all_prompts:
                category_counts[p.category] = category_counts.get(p.category, 0) + 1
            
            st.subheader("📊 Prompts by Category")
            category_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
            fig = px.pie(category_df, values='Count', names='Category', title="Distribution of Prompt Categories")
            st.plotly_chart(fig, use_container_width=True)
        
        # Export/Import controls
        st.subheader("💾 Library Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Library"):
                filename = st.session_state.generator.save_prompt_library()
                st.success(f"Library exported to {filename}")
        
        with col2:
            if st.button("📤 Load Library"):
                if st.session_state.generator.load_prompt_library():
                    st.success("Library loaded successfully!")
                    st.rerun()
                else:
                    st.warning("No library file found.")
    
    with tab5:
        st.header("🔬 Live Prompt Testing")
        
        # Custom prompt testing
        st.subheader("Test Custom Prompt")
        custom_prompt = st.text_area("Enter your baseball prompt:", height=100)
        
        if st.button("🧪 Test Custom Prompt") and custom_prompt:
            with st.spinner("Testing prompt..."):
                try:
                    response = st.session_state.generator.agent.process_query(custom_prompt)
                    quality = st.session_state.generator._evaluate_response_quality(response, None)
                    
                    st.write("**Response:**")
                    st.write(response)
                    st.write(f"**Estimated Quality Score**: {quality:.2f}")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Random prompt testing
        st.subheader("Test Random Prompt")
        if st.button("🎲 Generate & Test Random Prompt"):
            random_prompts = st.session_state.generator.generate_prompt_variations(1)
            if random_prompts:
                random_prompt = random_prompts[0]
                st.write(f"**Generated Prompt**: {random_prompt.prompt}")
                
                with st.spinner("Testing random prompt..."):
                    try:
                        response = st.session_state.generator.agent.process_query(random_prompt.prompt)
                        quality = st.session_state.generator._evaluate_response_quality(response, random_prompt)
                        
                        st.write("**Response:**")
                        st.write(response)
                        st.write(f"**Quality Score**: {quality:.2f}")
                        st.write(f"**Category**: {random_prompt.category}")
                        st.write(f"**Complexity**: {random_prompt.complexity}")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
