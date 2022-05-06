class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


class Solution(object):
    def preorder_traversal(self, root):
        """
        preorder traversal by recursion
        :param root:
        :return: output[]   preorder traversal result
        """
        if root is None:
            return []
        output = []
        output.extend(root.val)
        output.extend(Solution.preorder_traversal(self, root.left))
        output.extend(Solution.preorder_traversal(self, root.right))
        return output

    def inorder_traversal(self, root):
        """
        inorder traversal by recursion
        :param root:
        :return: output[]  inorder traversal result
        """
        if root is None:
            return []
        output = []
        output.extend(Solution.inorder_traversal(self, root.left))
        output.extend(root.val)
        output.extend(Solution.inorder_traversal(self, root.right))
        return output

    def postorder_traversal(self, root):
        """
        postorder traversal recursion
        :param root:
        :return: output[]  postorder traversal result
        """
        if root is None:
            return []
        output = []
        output.extend(Solution.postorder_traversal(self, root.left))
        output.extend(Solution.postorder_traversal(self, root.right))
        output.extend(root.val)
        return output

    @staticmethod
    def preorder_traversal1(root):
        """
        preorder traversal by iteration
        :param root:
        :return: output[]   preorder traversal result
        """
        if root is None:
            return []
        output = []
        stack = [root]
        while stack:
            root = stack.pop()
            if root.right:
                stack.append(root.right)
            if root.left:
                stack.append(root.left)
            output.append(root.val)
        return output

    @staticmethod
    def inorder_traversal1(root):
        """
        inorder traversal by iteration
        :param root:
        :return: output[]  inorder traversal result
        """
        if root is None:
            return []
        tree_type = type(root)
        stack = [root]
        output = []
        while stack:
            root = stack.pop()
            if type(root) != tree_type:
                output.append(root)
                continue
            if root.right:
                stack.append(root.right)
            stack.append(root.val)
            if root.left:
                stack.append(root.left)
        return output

    @staticmethod
    def postorder_traversal1(root):
        """
        postorder traversal by iteration
        :param root:
        :return: output[]  postorder traversal result
        """
        if root is None:
            return []
        output = []
        stack = [root]
        tree_type = type(root)
        while stack:
            root = stack.pop()
            if type(root) != tree_type:
                output.append(root)
                continue
            stack.append(root.val)
            if root.right:
                stack.append(root.right)
            if root.left:
                stack.append(root.left)
        return output


if __name__ == '__main__':
    node1 = TreeNode('F')
    node2 = TreeNode('B')
    node3 = TreeNode('G')
    node4 = TreeNode('A')
    node5 = TreeNode('D')
    node6 = TreeNode('C')
    node7 = TreeNode('E')
    node8 = TreeNode('I')
    node9 = TreeNode('H')
    node1.left = node2
    node1.right = node3
    node2.left = node4
    node2.right = node5
    node5.left = node6
    node5.right = node7
    node3.right = node8
    node8.left = node9
    sol = Solution()
    print(sol.preorder_traversal(node1), sol.inorder_traversal(node1), sol.postorder_traversal(node1))
    print(sol.preorder_traversal1(node1), sol.inorder_traversal1(node1), sol.postorder_traversal1(node1))
