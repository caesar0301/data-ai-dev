#!/usr/bin/env python3
"""
Test script for NeuG basic graph functionality
Tests: database creation, data loading, Cypher queries, graph analytics
"""

import neug
import os
import tempfile
import shutil

def test_basic_graph_functionality():
    """Test basic graph database operations"""

    print("=" * 70)
    print("NeuG Basic Graph Functionality Test")
    print("=" * 70)

    # Create a temporary directory for the database
    db_path = tempfile.mkdtemp(prefix="neug_test_")
    print(f"\n[1] Creating database at: {db_path}")

    try:
        # Step 1: Create database
        db = neug.Database(db_path)
        print("✓ Database created successfully")

        # Step 2: Load built-in dataset
        print("\n[2] Loading built-in 'tinysnb' dataset...")
        db.load_builtin_dataset("tinysnb")
        print("✓ Dataset loaded successfully")

        # Step 3: Create connection
        print("\n[3] Creating connection...")
        conn = db.connect()
        print("✓ Connection established")

        # Step 4: Run basic Cypher query - find person relationships
        print("\n[4] Testing Cypher query: Find person-knows-person relationships...")
        query1 = """
            MATCH (a:person)-[:knows]->(b:person)
            RETURN a.fName, b.fName
            ORDER BY a.fName, b.fName
            LIMIT 5
        """
        result1 = conn.execute(query1)
        print(f"✓ Query executed successfully, found {len(result1)} relationships")
        count = 0
        for record in result1:
            if count >= 5:
                break
            print(f"   {count+1}. {record[0]} knows {record[1]}")
            count += 1

        # Step 5: Run triangle detection query
        print("\n[5] Testing triangle detection query...")
        query2 = """
            MATCH (a:person)-[:knows]->(b:person)-[:knows]->(c:person),
                  (a)-[:knows]->(c)
            RETURN a.fName, b.fName, c.fName
        """
        result2 = conn.execute(query2)
        print(f"✓ Triangle query executed successfully, found {len(result2)} triangles")
        if len(result2) > 0:
            print("   Sample triangles found:")
            count = 0
            for record in result2:
                if count >= 3:
                    break
                print(f"   {count+1}. {record[0]}, {record[1]}, {record[2]} are mutual friends")
                count += 1

        # Step 6: Test aggregation query
        print("\n[6] Testing aggregation: Count nodes by label...")
        query3 = """
            MATCH (n)
            RETURN labels(n) as label, count(n) as count
            ORDER BY count DESC
        """
        result3 = conn.execute(query3)
        print(f"✓ Aggregation query executed successfully")
        for record in result3:
            print(f"   Label {record[0]}: {record[1]} nodes")

        # Step 7: Test filtering query
        print("\n[7] Testing filtering: Find persons with specific conditions...")
        query4 = """
            MATCH (p:person)
            WHERE p.age > 30
            RETURN p.fName, p.age, p.gender
            ORDER BY p.age DESC
            LIMIT 3
        """
        result4 = conn.execute(query4)
        print(f"✓ Filter query executed successfully, found {len(result4)} persons")
        count = 0
        for record in result4:
            print(f"   {count+1}. {record[0]} (age: {record[1]}, gender: {record[2]})")
            count += 1

        # Step 8: Close connection
        print("\n[8] Closing connection...")
        conn.close()
        print("✓ Connection closed")

        print("\n" + "=" * 70)
        print("All basic graph functionality tests PASSED ✓")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print(f"\n[Cleanup] Removing test database...")
        shutil.rmtree(db_path, ignore_errors=True)
        print("✓ Cleanup completed")

    return True

if __name__ == "__main__":
    success = test_basic_graph_functionality()
    exit(0 if success else 1)